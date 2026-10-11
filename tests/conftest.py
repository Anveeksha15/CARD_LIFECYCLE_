import pytest

from ml.generate_data import generate
from ml.train import train
from scripts.seed_issuer import main as seed_issuer
from services.gateway.db import init_db as init_gateway_db
from services.issuer.accounts import init_db as init_issuer_db


@pytest.fixture(scope="session", autouse=True)
def _init_dbs():
    init_gateway_db()
    init_issuer_db()
    seed_issuer()


@pytest.fixture(scope="session")
def fraud_model_prefix(tmp_path_factory):
    """A real fraud model trained once per test session (~3 s). 20k rows so the validation
    slice is guaranteed to contain attacks."""
    out = tmp_path_factory.mktemp("fraud_artifacts")
    train(generate(n=20_000, seed=7), out_dir=out)
    return out / "gbm"
