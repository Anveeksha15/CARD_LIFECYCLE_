# Rules baseline

Dataset: `python -m ml.generate_data` (seed 42, 99,102 rows). Synthetic data only.
Test slice = last 30% of the timeline (same split the models use).

| slice | precision | recall | F1 | FPR (normal) | campaign recall |
|---|---|---|---|---|---|
| all | 0.820 | 0.469 | 0.596 | 0.0038 | 0.655 |
| test | 0.569 | 0.515 | 0.541 | 0.0124 | 0.800 |

| scenario | share flagged |
|---|---|
| attack_burst | 0.923 |
| attack_rotating_identity | 0.004 |
| attack_rotating_ip | 0.003 |
| attack_slow | 0.176 |
| flash_sale | 0.298 |
| normal | 0.000 |
| normal_retry | 0.003 |
| normal_shared_ip | 0.000 |
