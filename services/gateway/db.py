from datetime import UTC, datetime

from sqlmodel import Field, Session, SQLModel, create_engine

_engine = create_engine("sqlite:///gateway.db", connect_args={"check_same_thread": False})


class Token(SQLModel, table=True):
    token: str = Field(primary_key=True)
    card_number: str


class PaymentIntent(SQLModel, table=True):
    id: str = Field(primary_key=True)
    amount_cents: int
    token: str
    status: str
    decline_reason: str | None = None
    request_id: str | None = None
    settled: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FraudCheck(SQLModel, table=True):
    """One row per payment the fraud service actually scored. Its own table rather than
    columns on PaymentIntent, because create_all() adds missing tables but never alters
    existing ones."""

    intent_id: str = Field(primary_key=True)
    risk: float
    decision: str
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def init_db() -> None:
    SQLModel.metadata.create_all(_engine)


def get_session() -> Session:
    return Session(_engine)
