from services.fraud.rules import rules_flag, rules_score

QUIET = {"attempts_per_ip_60s": 1, "distinct_cards_per_ip_5m": 1, "decline_rate_ip_5m": 0.0}


def test_quiet_traffic_scores_zero():
    assert rules_score(QUIET) == 0.0
    assert not rules_flag(QUIET)


def test_thresholds_are_strict():
    at_limit = {"attempts_per_ip_60s": 10, "distinct_cards_per_ip_5m": 5, "decline_rate_ip_5m": 0.6}
    assert rules_score(at_limit) == 0.0


def test_attempt_rule_alone_flags():
    f = {**QUIET, "attempts_per_ip_60s": 11}
    assert rules_score(f) == 0.5
    assert rules_flag(f)


def test_cards_and_declines_together_flag():
    f = {**QUIET, "distinct_cards_per_ip_5m": 6, "decline_rate_ip_5m": 0.9}
    assert rules_score(f) == 0.5
    assert rules_flag(f)


def test_single_weak_rule_does_not_flag():
    assert not rules_flag({**QUIET, "distinct_cards_per_ip_5m": 6})
    assert not rules_flag({**QUIET, "decline_rate_ip_5m": 0.9})


def test_score_capped_at_one():
    f = {"attempts_per_ip_60s": 500, "distinct_cards_per_ip_5m": 500, "decline_rate_ip_5m": 1.0}
    assert rules_score(f) == 1.0
