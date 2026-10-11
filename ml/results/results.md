# Rules vs models (test slice)

Synthetic data only. Test slice = last 30% of the timeline: 30,472 events, 939 attack events in 15 campaigns.
Model thresholds were chosen on the validation slice (60-70%), never on test.

| method | precision | recall | F1 | FPR (normal) | campaign recall | PR-AUC |
|---|---|---|---|---|---|---|
| rules | 0.569 | 0.515 | 0.541 | 0.0124 | 0.800 | - |
| logreg | 0.659 | 0.934 | 0.773 | 0.0154 | 1.000 | 0.735 |
| gbm | 0.962 | 0.944 | 0.953 | 0.0012 | 1.000 | 0.974 |

## Share flagged by scenario

Attack rows: higher is better. Normal rows: lower is better.

| scenario | rules | logreg | gbm |
|---|---|---|---|
| attack_burst | 0.912 | 0.981 | 0.985 |
| attack_rotating_identity | 0.021 | 0.766 | 0.883 |
| attack_rotating_ip | 0.000 | 0.941 | 0.941 |
| attack_slow | 0.191 | 0.900 | 0.883 |
| flash_sale | 0.298 | 0.337 | 0.014 |
| normal | 0.000 | 0.000 | 0.000 |
| normal_retry | 0.009 | 0.021 | 0.002 |
| normal_shared_ip | 0.000 | 0.020 | 0.005 |

## Error analysis (gbm)

Least-caught attack campaigns:

| campaign | scenario | events | share caught |
|---|---|---|---|
| c0046 | attack_slow | 42 | 0.79 |
| c0052 | attack_slow | 58 | 0.83 |
| c0053 | attack_rotating_identity | 94 | 0.88 |
| c0050 | attack_slow | 55 | 0.91 |
| c0006 | attack_rotating_ip | 44 | 0.93 |
| c0038 | attack_rotating_ip | 54 | 0.94 |
| c0027 | attack_rotating_ip | 37 | 0.95 |
| c0009 | attack_burst | 23 | 0.96 |

False positives by scenario:

- flash_sale: 17
- normal: 10
- normal_shared_ip: 6
- normal_retry: 2

Most-flagged normal IPs:

- 23.197.167.112: 4
- 58.142.167.141: 4
- 110.219.22.178: 3
- 187.194.183.200: 2
- 216.143.37.142: 1

## gbm permutation importance (drop in PR-AUC on test)

| feature | importance |
|---|---|
| distinct_cards_per_ip_5m | 0.2843 |
| attempts_per_email_60s | 0.2110 |
| decline_rate_ip_5m | 0.0667 |
| amount_cents | 0.0562 |
| attempts_per_bin_60s | 0.0100 |
| attempts_per_ip_60s | 0.0039 |
| failed_per_ip_60s | 0.0010 |
