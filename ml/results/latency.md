# Fraud-check latency

Docker Desktop on Windows, all services on one machine; client calls via 127.0.0.1.
Produced by `python -m scripts.fraud_latency` with FRAUD_CHECK_ENABLED on, then off.

| | gateway p50 off | gateway p50 on | fraud check adds (p50) | adds (p95) |
|---|---|---|---|---|
| before pooling (Day 5) | 42.7 ms | 60.1 ms | +17.4 ms | +18.9 ms |
| after pooling (Day 6) | 20.0 ms | 28.3 ms | +8.3 ms | +7.5 ms |

Raw runs:

**Day 5, before connection pooling**
_run 2026-10-11 12:21, fraud check on_

| measurement | n | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| /v1/score (direct) | 500 | 5.6 | 8.4 | 17.8 | 38.9 |
| gateway payment intent, fraud check on | 200 | 60.1 | 83.6 | 94.6 | 110.5 |

_run 2026-10-11 12:22, fraud check off_

| measurement | n | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| /v1/score (direct) | 100 | 5.7 | 7.7 | 15.5 | 15.6 |
| gateway payment intent, fraud check off | 200 | 42.7 | 64.7 | 85.8 | 121.2 |

**Day 6, after connection pooling (common/http.py)**
_run 2026-10-11 12:30, fraud check on_

| measurement | n | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| /v1/score (direct) | 500 | 4.8 | 6.4 | 7.5 | 20.8 |
| gateway payment intent, fraud check on | 200 | 28.3 | 34.5 | 64.7 | 202.6 |

_run 2026-10-11 12:30, fraud check off_

| measurement | n | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| /v1/score (direct) | 100 | 6.5 | 10.9 | 45.4 | 45.6 |
| gateway payment intent, fraud check off | 200 | 20.0 | 27.0 | 32.8 | 34.4 |
