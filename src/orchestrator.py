"""LLM orchestrator -- calls OpenAI and Anthropic APIs directly."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.cost.model_router import ModelRouter
from src.cost.rate_limiter import RateLimiter
from src.cost.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float


class TalonOrchestrator:
    def __init__(self, model_config, cost_controls, token_tracker, rate_limiter,
                 router: ModelRouter | None = None):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.tracker = token_tracker
        self.rate_limiter = rate_limiter
        self.router = router if router is not None else ModelRouter(
            model_config, cost_controls, token_tracker
        )

    def call_model(self, task_type, prompt, system_prompt=None, response_format=None):
        model_config = self.router.get_model_for_task(task_type)
        if model_config is None:
            logger.warning("No model available for task %s (budget exhausted).", task_type)
            return None
        if not self.router.can_afford(model_config):
            logger.warning("Cannot afford model call for task %s.", task_type)
            return None

        provider = model_config.get("provider", "openai")
        self.rate_limiter.acquire_sync(provider)

        model_name = model_config["model"]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_tokens = model_config.get("max_tokens", 500)
        temperature = model_config.get("temperature", 0.3)

        if model_name.startswith("anthropic/"):
            result = self._call_anthropic(model_name, messages, max_tokens, temperature)
        else:
            result = self._call_openai(model_name, messages, max_tokens, temperature)

        if result is None:
            return None

        cost = self.tracker.estimate_cost(model_config, result.tokens_input, result.tokens_output)
        self.tracker.log_usage(model_name, task_type, result.tokens_input, result.tokens_output, cost)
        result.cost_usd = cost

        logger.info("LLM call: task=%s model=%s tokens=%d+%d cost=$%.4f",
                     task_type, model_name, result.tokens_input, result.tokens_output, cost)
        return result

    def _call_openai(self, model_name, messages, max_tokens, temperature):
        if not self.openai_client:
            logger.error("No OpenAI API key configured.")
            return None
        model_id = model_name.replace("openai/", "")
        try:
            response = self.openai_client.chat.completions.create(
                **self._openai_kwargs(model_id, messages, max_tokens, temperature)
            )
            choice = response.choices[0] if response.choices else None
            usage = response.usage
            return LLMResponse(
                content=choice.message.content if choice else "",
                model=model_name,
                tokens_input=usage.prompt_tokens if usage else 0,
                tokens_output=usage.completion_tokens if usage else 0,
                cost_usd=0,
            )
        except Exception:
            logger.exception("OpenAI call failed for %s", model_name)
            return None

    @staticmethod
    def _is_openai_reasoning(model_id: str) -> bool:
        """The gpt-5 and o-series reasoning models require
        `max_completion_tokens` and only accept the default temperature."""
        m = model_id.lower()
        return m.startswith(("gpt-5", "o1", "o3", "o4"))

    def _openai_kwargs(self, model_id, messages, max_tokens, temperature) -> dict:
        """Build chat.completions kwargs, adapting to reasoning models."""
        kwargs: dict = {"model": model_id, "messages": messages}
        if self._is_openai_reasoning(model_id):
            # Reasoning models: max_completion_tokens, no custom temperature.
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            if temperature is not None and temperature > 0:
                kwargs["temperature"] = temperature
        return kwargs

    def _call_anthropic(self, model_name, messages, max_tokens, temperature):
        if not ANTHROPIC_API_KEY:
            logger.error("No Anthropic API key configured.")
            return None
        # Aliases like "claude-sonnet-4-6" and "claude-opus-4-6" are themselves
        # valid Anthropic model IDs; pass through unchanged. The previous
        # remapping pointed to dated snapshots (claude-*-4-20250514) that were
        # deprecated April 2026 with retirement June 15, 2026.
        anthropic_model = model_name.replace("anthropic/", "")

        system_text = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                user_messages.append(m)

        body = {
            "model": anthropic_model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if system_text:
            body["system"] = system_text
        # Opus 4.x deprecated the temperature parameter and the API rejects it
        # with HTTP 400. Only send temperature for non-Opus models.
        if temperature > 0 and "opus" not in anthropic_model:
            body["temperature"] = temperature

        try:
            with httpx.Client(timeout=90) as http:
                resp = http.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=body,
                )
                data = resp.json()

            if "content" not in data:
                logger.error("Anthropic error: %s", data.get("error", data))
                return None

            content = data["content"][0]["text"] if data["content"] else ""
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                model=model_name,
                tokens_input=usage.get("input_tokens", 0),
                tokens_output=usage.get("output_tokens", 0),
                cost_usd=0,
            )
        except Exception:
            logger.exception("Anthropic call failed for %s", model_name)
            return None

    def call_model_vision(self, task_type, text_prompt, image_b64_png,
                          system_prompt=None):
        """Vision-capable model call. Supports Anthropic (image content
        block) and OpenAI (image_url with data URL). Returns LLMResponse
        or None on failure / budget exhaustion."""
        model_config = self.router.get_model_for_task(task_type)
        if model_config is None:
            return None
        if not self.router.can_afford(model_config, estimated_tokens=2000):
            return None

        provider = model_config.get("provider", "openai")
        self.rate_limiter.acquire_sync(provider)

        model_name = model_config["model"]
        max_tokens = model_config.get("max_tokens", 800)
        temperature = model_config.get("temperature", 0.2)

        if model_name.startswith("anthropic/"):
            result = self._call_anthropic_vision(
                model_name, system_prompt, text_prompt, image_b64_png,
                max_tokens, temperature,
            )
        else:
            result = self._call_openai_vision(
                model_name, system_prompt, text_prompt, image_b64_png,
                max_tokens, temperature,
            )

        if result is None:
            return None

        cost = self.tracker.estimate_cost(model_config, result.tokens_input, result.tokens_output)
        self.tracker.log_usage(model_name, task_type, result.tokens_input, result.tokens_output, cost)
        result.cost_usd = cost
        logger.info(
            "Vision call: task=%s model=%s tokens=%d+%d cost=$%.4f",
            task_type, model_name, result.tokens_input, result.tokens_output, cost,
        )
        return result

    def _call_anthropic_vision(self, model_name, system_prompt, text_prompt,
                               image_b64_png, max_tokens, temperature):
        if not ANTHROPIC_API_KEY:
            logger.error("No Anthropic API key configured.")
            return None
        anthropic_model = model_name.replace("anthropic/", "")
        body = {
            "model": anthropic_model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64_png,
                            },
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
        }
        if system_prompt:
            body["system"] = system_prompt
        # Opus 4.x rejects the temperature parameter (HTTP 400); omit it.
        if temperature > 0 and "opus" not in anthropic_model:
            body["temperature"] = temperature
        try:
            with httpx.Client(timeout=120) as http:
                resp = http.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=body,
                )
                data = resp.json()
            if "content" not in data:
                logger.error("Anthropic vision error: %s", data.get("error", data))
                return None
            content = data["content"][0]["text"] if data["content"] else ""
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                model=model_name,
                tokens_input=usage.get("input_tokens", 0),
                tokens_output=usage.get("output_tokens", 0),
                cost_usd=0,
            )
        except Exception:
            logger.exception("Anthropic vision call failed for %s", model_name)
            return None

    def _call_openai_vision(self, model_name, system_prompt, text_prompt,
                            image_b64_png, max_tokens, temperature):
        if not self.openai_client:
            logger.error("No OpenAI API key configured.")
            return None
        model_id = model_name.replace("openai/", "")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64_png}",
                    },
                },
            ],
        })
        try:
            response = self.openai_client.chat.completions.create(
                **self._openai_kwargs(model_id, messages, max_tokens, temperature)
            )
            choice = response.choices[0] if response.choices else None
            usage = response.usage
            return LLMResponse(
                content=choice.message.content if choice else "",
                model=model_name,
                tokens_input=usage.prompt_tokens if usage else 0,
                tokens_output=usage.completion_tokens if usage else 0,
                cost_usd=0,
            )
        except Exception:
            logger.exception("OpenAI vision call failed for %s", model_name)
            return None

    def call_model_json(self, task_type, prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = "Respond with valid JSON only. No markdown, no code fences."
        result = self.call_model(task_type, prompt, system_prompt)
        if result is None:
            return None
        text = result.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM JSON: %s", text[:200])
            return None
