from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_type = Column(String, nullable=False)
    started_at = Column(String, nullable=False)
    completed_at = Column(String)
    tickers_scanned = Column(Text)  # JSON array
    setups_found = Column(Integer, default=0)
    alerts_sent = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)

    setups = relationship("Setup", back_populates="scan")


class Setup(Base):
    __tablename__ = "setups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey("scans.id"))
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    setup_type = Column(String, nullable=False)
    timeframe = Column(String)
    detected_at = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    score_breakdown = Column(Text)  # JSON
    raw_signals = Column(Text)  # JSON
    sources_used = Column(Text)  # JSON array
    promoted_to_alert = Column(Boolean, default=False)

    scan = relationship("Scan", back_populates="setups")
    alert = relationship("Alert", back_populates="setup", uselist=False)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    setup_id = Column(Integer, ForeignKey("setups.id"))
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    contract = Column(String, nullable=False)
    entry_price_low = Column(Float)
    entry_price_high = Column(Float)
    target_price = Column(Float)
    stop_price = Column(Float)
    confidence = Column(Integer)
    rationale = Column(Text)
    sent_at = Column(String, nullable=False)
    telegram_message_id = Column(Integer)
    mode = Column(String, nullable=False)

    setup = relationship("Setup", back_populates="alert")
    outcome = relationship("Outcome", back_populates="alert", uselist=False)
    feedback_entries = relationship("Feedback", back_populates="alert")


class Outcome(Base):
    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    entry_filled = Column(Boolean, default=False)
    actual_entry_price = Column(Float)
    actual_exit_price = Column(Float)
    exit_reason = Column(String)
    pnl_dollars = Column(Float)
    pnl_pct = Column(Float)
    max_favorable = Column(Float)
    max_adverse = Column(Float)
    hold_duration_minutes = Column(Integer)
    resolved_at = Column(String)
    notes = Column(Text)

    alert = relationship("Alert", back_populates="outcome")


class CounterfactualOutcome(Base):
    """Shadow outcome for a near-miss setup that was scored but NOT alerted
    (filtered by score floor, confidence gate, IV-reject, or vision PASS).

    Tracked separately from real Outcomes so primary win-rate stats stay
    clean, but resolved by the same intraday poll. Used to learn whether the
    alerting gates are too strict (lots of filtered setups that would have won)
    or correctly conservative.
    """

    __tablename__ = "counterfactual_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    setup_id = Column(Integer, ForeignKey("setups.id"))
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    contract = Column(String, nullable=False)
    filter_reason = Column(String)  # why it wasn't alerted
    score = Column(Float)
    entry_price = Column(Float)
    target_price = Column(Float)
    stop_price = Column(Float)
    actual_exit_price = Column(Float)
    exit_reason = Column(String)
    pnl_pct = Column(Float)
    max_favorable = Column(Float)
    max_adverse = Column(Float)
    created_at = Column(String, nullable=False)
    resolved_at = Column(String)


class SourceScore(Base):
    __tablename__ = "source_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String, nullable=False)
    source_subtype = Column(String)
    total_signals = Column(Integer, default=0)
    winning_signals = Column(Integer, default=0)
    avg_pnl_pct = Column(Float, default=0.0)
    reliability_score = Column(Float, default=0.50)
    enabled = Column(Boolean, default=True)
    disabled_reason = Column(String)
    last_updated = Column(String)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    feedback_type = Column(String, nullable=False)
    user_note = Column(Text)
    received_at = Column(String, nullable=False)

    alert = relationship("Alert", back_populates="feedback_entries")


class CostLog(Base):
    __tablename__ = "cost_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False)
    model = Column(String, nullable=False)
    task = Column(String, nullable=False)
    tokens_input = Column(Integer)
    tokens_output = Column(Integer)
    cost_usd = Column(Float)
    logged_at = Column(String, nullable=False)


class PerformanceReview(Base):
    __tablename__ = "performance_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    period_type = Column(String, nullable=False)
    period_start = Column(String, nullable=False)
    period_end = Column(String, nullable=False)
    total_alerts = Column(Integer)
    winning_alerts = Column(Integer)
    losing_alerts = Column(Integer)
    total_pnl_pct = Column(Float)
    avg_pnl_pct = Column(Float)
    best_trade = Column(Text)  # JSON
    worst_trade = Column(Text)  # JSON
    top_sources = Column(Text)  # JSON
    bottom_sources = Column(Text)  # JSON
    lessons = Column(Text)
    source_score_updates = Column(Text)  # JSON
    created_at = Column(String, nullable=False)
