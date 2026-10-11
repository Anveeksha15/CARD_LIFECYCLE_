import fakeredis
import pytest

from services.fraud.features import Tx, observe, record_outcome


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


def tx(event_id, card="c1", ip="1.1.1.1", email="a@x", bin_="424242"):
    return Tx(event_id=event_id, card_token=card, bin=bin_, ip=ip, email=email)


def test_first_event_counts_only_itself(r):
    f = observe(r, tx("e1"), now=0.0)
    assert f == {
        "attempts_per_ip_60s": 1,
        "failed_per_ip_60s": 0,
        "distinct_cards_per_ip_5m": 1,
        "decline_rate_ip_5m": 0.0,
        "attempts_per_email_60s": 1,
        "attempts_per_bin_60s": 1,
    }


def test_outcome_only_visible_after_record_outcome(r):
    e1 = tx("e1")
    observe(r, e1, now=0.0)
    assert observe(r, tx("e2"), now=1.0)["failed_per_ip_60s"] == 0

    record_outcome(r, e1, declined=True)
    f = observe(r, tx("e3"), now=2.0)
    assert f["failed_per_ip_60s"] == 1
    assert f["decline_rate_ip_5m"] == pytest.approx(1 / 2)  # 1 decline over e1, e2


def test_approved_outcome_records_nothing(r):
    e1 = tx("e1")
    observe(r, e1, now=0.0)
    record_outcome(r, e1, declined=False)
    assert observe(r, tx("e2"), now=1.0)["decline_rate_ip_5m"] == 0.0


def test_short_window_boundary_is_exclusive(r):
    e1 = tx("e1")
    observe(r, e1, now=0.0)
    record_outcome(r, e1, declined=True)
    f = observe(r, tx("e2"), now=60.0)
    assert f["attempts_per_ip_60s"] == 1  # e1 is exactly 60s old: out of the 60s window
    assert f["failed_per_ip_60s"] == 0
    assert f["attempts_per_email_60s"] == 1
    assert f["attempts_per_bin_60s"] == 1
    assert f["decline_rate_ip_5m"] == 1.0  # ...but still inside the 5-minute window


def test_cards_counted_once_and_expire(r):
    observe(r, tx("e1", card="c1"), now=0.0)
    observe(r, tx("e2", card="c2"), now=10.0)
    assert observe(r, tx("e3", card="c1"), now=250.0)["distinct_cards_per_ip_5m"] == 2
    # c2 was last seen at t=10 (expired); c1 was refreshed at t=250.
    assert observe(r, tx("e4", card="c3"), now=320.0)["distinct_cards_per_ip_5m"] == 2


def test_keys_are_isolated(r):
    observe(r, tx("e1"), now=0.0)
    f = observe(r, tx("e2", ip="2.2.2.2", email="b@x", bin_="555555"), now=1.0)
    assert f["attempts_per_ip_60s"] == 1
    assert f["attempts_per_email_60s"] == 1
    assert f["attempts_per_bin_60s"] == 1


def test_shared_email_across_ips(r):
    observe(r, tx("e1", ip="1.1.1.1"), now=0.0)
    f = observe(r, tx("e2", ip="2.2.2.2"), now=1.0)
    assert f["attempts_per_ip_60s"] == 1
    assert f["attempts_per_email_60s"] == 2
    assert f["attempts_per_bin_60s"] == 2


def test_keys_get_ttls(r):
    e1 = tx("e1")
    observe(r, e1, now=0.0)
    record_outcome(r, e1, declined=True)
    keys = list(r.scan_iter("fraud:*"))
    assert len(keys) == 5
    assert all(r.ttl(k) > 0 for k in keys)


def test_prefixes_do_not_interfere(r):
    observe(r, tx("e1"), now=0.0, prefix="a")
    assert observe(r, tx("e2"), now=1.0, prefix="b")["attempts_per_ip_60s"] == 1


def test_decline_is_stamped_with_the_attempt_time(r):
    e1 = tx("e1")
    observe(r, e1, now=0.0)
    observe(r, tx("e2"), now=100.0)  # e2's outcome hasn't arrived yet
    record_outcome(r, e1, declined=True)  # e1's outcome arrives late
    # At t=300.5 e1 has left the 5-minute window - its decline must leave with it, and the
    # rate must never exceed 1.
    f = observe(r, tx("e3"), now=300.5)
    assert f["decline_rate_ip_5m"] == 0.0


def test_outcome_for_unknown_attempt_is_ignored(r):
    record_outcome(r, tx("never_observed"), declined=True)
    assert observe(r, tx("e1"), now=0.0)["decline_rate_ip_5m"] == 0.0
