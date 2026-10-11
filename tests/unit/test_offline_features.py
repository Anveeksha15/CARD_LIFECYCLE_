import pandas as pd
import pytest

from ml.offline_features import compute_features


def _events(rows):
    return pd.DataFrame(rows, columns=["ts", "card_token", "bin", "ip", "email", "outcome"])


@pytest.fixture
def features():
    df = _events(
        [
            (0.0, "c1", "424242", "1.1.1.1", "a@x", "declined"),
            (10.0, "c2", "424242", "1.1.1.1", "a@x", "declined"),
            (20.0, "c3", "424242", "1.1.1.1", "a@x", "approved"),
            (20.0, "c9", "555555", "2.2.2.2", "b@x", "approved"),  # other ip/email/bin
            (60.0, "c3", "424242", "1.1.1.1", "a@x", "declined"),  # event at t=0 drops out
            (400.0, "c4", "424242", "1.1.1.1", "a@x", "approved"),  # everything expired
        ]
    )
    return compute_features(df)


def test_first_event_counts_only_itself(features):
    first = features.iloc[0]
    assert first["attempts_per_ip_60s"] == 1
    assert first["failed_per_ip_60s"] == 0
    assert first["distinct_cards_per_ip_5m"] == 1
    assert first["decline_rate_ip_5m"] == 0.0


def test_current_outcome_is_not_used(features):
    # Event 2 is itself declined, but only event 1's decline is known when it's scored.
    second = features.iloc[1]
    assert second["failed_per_ip_60s"] == 1
    assert second["decline_rate_ip_5m"] == 1.0


def test_keys_are_isolated(features):
    other = features.iloc[3]
    assert other["attempts_per_ip_60s"] == 1
    assert other["attempts_per_email_60s"] == 1
    assert other["attempts_per_bin_60s"] == 1


def test_short_window_boundary_is_exclusive(features):
    # At t=60 the t=0 event is exactly 60s old and falls out of the 60s window,
    # but is still inside the 5-minute window.
    row = features.iloc[4]
    assert row["attempts_per_ip_60s"] == 3  # t=10, t=20, and itself
    assert row["failed_per_ip_60s"] == 1  # only t=10
    assert row["attempts_per_email_60s"] == 3
    assert row["distinct_cards_per_ip_5m"] == 3  # c1, c2, c3 (repeat card counted once)
    assert row["decline_rate_ip_5m"] == pytest.approx(2 / 3)


def test_long_window_expiry(features):
    last = features.iloc[5]
    assert last["attempts_per_ip_60s"] == 1
    assert last["distinct_cards_per_ip_5m"] == 1
    assert last["decline_rate_ip_5m"] == 0.0
