import pandas as pd
import pytest

from ml.generate_data import COLUMNS, generate


@pytest.fixture(scope="module")
def small():
    return generate(n=3000, seed=7)


def test_same_seed_is_reproducible(small):
    pd.testing.assert_frame_equal(small, generate(n=3000, seed=7))


def test_different_seed_differs(small):
    assert not small.equals(generate(n=3000, seed=8))


def test_schema_and_ordering(small):
    assert list(small.columns) == COLUMNS
    assert small["ts"].is_monotonic_increasing
    assert small["event_id"].is_unique
    assert set(small["outcome"]) == {"approved", "declined"}


def test_attack_share_is_realistic(small):
    assert 0.01 <= small["is_attack"].mean() <= 0.08


def test_attack_rows_belong_to_campaigns(small):
    attacks = small[small["is_attack"] == 1]
    assert (attacks["campaign_id"] != "").all()
    assert (small.loc[small["is_attack"] == 0, "campaign_id"] == "").all()


def test_no_card_numbers_anywhere(small):
    pan_like = small.astype(str).apply(lambda col: col.str.fullmatch(r"\d{13,19}"))
    assert not pan_like.to_numpy().any()


def test_amounts_overlap_between_classes(small):
    normal = small.loc[small["is_attack"] == 0, "amount_cents"]
    attack = small.loc[small["is_attack"] == 1, "amount_cents"]
    assert (normal < 500).any()
    assert (attack > 500).any()
