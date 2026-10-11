"""Baseline rule-based fraud scorer - the number every model has to beat."""

MAX_ATTEMPTS_PER_IP_60S = 10
MAX_DISTINCT_CARDS_PER_IP_5M = 5
MAX_DECLINE_RATE_IP_5M = 0.6

FLAG_THRESHOLD = 0.5


def rules_score(f: dict) -> float:
    score = 0.0
    if f["attempts_per_ip_60s"] > MAX_ATTEMPTS_PER_IP_60S:
        score += 0.5
    if f["distinct_cards_per_ip_5m"] > MAX_DISTINCT_CARDS_PER_IP_5M:
        score += 0.3
    if f["decline_rate_ip_5m"] > MAX_DECLINE_RATE_IP_5M:
        score += 0.2
    return min(score, 1.0)


def rules_flag(f: dict, threshold: float = FLAG_THRESHOLD) -> bool:
    return rules_score(f) >= threshold
