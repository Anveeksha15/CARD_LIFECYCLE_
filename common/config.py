from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    webhook_secret: str = "change-me-dev-secret"
    settlement_delay_seconds: int = 5

    gateway_port: int = 8000
    acquirer_port: int = 8001
    processor_port: int = 8002
    network_port: int = 8003
    issuer_port: int = 8004
    fraud_port: int = 8005

    gateway_url: str = "http://localhost:8000"
    acquirer_url: str = "http://localhost:8001"
    processor_url: str = "http://localhost:8002"
    network_url: str = "http://localhost:8003"
    issuer_url: str = "http://localhost:8004"
    fraud_url: str = "http://localhost:8005"

    merchant_webhook_url: str = "http://localhost:9000/webhook"

    request_timeout_seconds: float = 1.0
    max_retry_attempts: int = 2
    retry_backoff_base_seconds: float = 0.2

    simulate_lost_response_delay_seconds: float = 3.0

    settlement_poll_interval_seconds: float = 1.0
    settlement_fee_bps: int = 290  # 2.9%, percentage-only - no flat per-transaction fee

    # Path prefix of a model saved by `python -m ml.train` ({prefix}.joblib + {prefix}.json).
    fraud_model_path: str = "ml/artifacts/gbm"
    fraud_review_threshold: float = 0.5  # block threshold comes from the model's metadata

    fraud_check_enabled: bool = False
    fraud_timeout_seconds: float = 0.15
    card_fingerprint_secret: str = "change-me-dev-fingerprint"


settings = Settings()
