# STM-BabyVision: Claude pilot

Dataset: `tmp\staticbench\v1`

## Overall

| run | items | correct | accuracy | macro accuracy (subtypes) | mean score | abstained | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-full | 37 | 31 | 83.8% | 87.1% | 0.862 | 0 | 0 | 0 |
| claude-image | 37 | 23 | 62.2% | 70.9% | 0.685 | 0 | 0 | 0 |
| baseline (classical) | 37 | 30 | 81.1% | 86.9% | 0.857 | 0 | 0 | 0 |

## Per subtype

Accuracy counts an abstention or a missing answer as wrong; *answered* is accuracy over the items that got a valid answer. Errors are over answered items, in the unit shown (position lists: mean distance of matched points).

| subtype | run | n | accuracy | answered acc. | abstain | error median | error p90 | extra |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tip.condition | claude-full | 7 | 42.9% | 42.9% | 0 | -  | - | recall blunt: 0.0%, double: 66.7%, unstable: 100.0% |
| tip.condition | claude-image | 7 | 42.9% | 42.9% | 0 | -  | - | recall blunt: 0.0%, double: 66.7%, unstable: 100.0% |
| tip.condition | baseline (classical) | 7 | 71.4% | 71.4% | 0 | -  | - | recall blunt: 100.0%, double: 33.3%, unstable: 100.0% |
| adatoms.count | claude-full | 10 | 90.0% | 90.0% | 0 | 0.000 atoms | 0.000 | within +-1: 100.0% |
| adatoms.count | claude-image | 10 | 100.0% | 100.0% | 0 | 0.000 atoms | 0.000 | within +-1: 100.0% |
| adatoms.count | baseline (classical) | 10 | 80.0% | 80.0% | 0 | 0.000 atoms | 1.000 | within +-1: 90.0% |
| adatoms.positions | claude-full | 10 | 90.0% | 90.0% | 0 | 0.043 nm | 0.059 | P 0.99 / R 0.99 / F1 0.99 |
| adatoms.positions | claude-image | 10 | 70.0% | 70.0% | 0 | 0.047 nm | 0.060 | P 0.93 / R 0.94 / F1 0.93 |
| adatoms.positions | baseline (classical) | 10 | 70.0% | 70.0% | 0 | 0.038 nm | 0.081 | P 0.99 / R 0.87 / F1 0.87 |
| drift.vector | claude-full | 1 | 100.0% | 100.0% | 0 | 0.006 nm | 0.006 |  |
| drift.vector | claude-image | 1 | 100.0% | 100.0% | 0 | 0.017 nm | 0.017 |  |
| drift.vector | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.023 nm | 0.023 |  |
| herringbone.period | claude-full | 8 | 100.0% | 100.0% | 0 | 0.027 nm | 0.050 |  |
| herringbone.period | claude-image | 8 | 12.5% | 12.5% | 0 | 1.175 nm | 2.314 |  |
| herringbone.period | baseline (classical) | 8 | 100.0% | 100.0% | 0 | 0.002 nm | 0.046 |  |
| tipchange.row | claude-full | 1 | 100.0% | 100.0% | 0 | 0.000 rows | 0.000 |  |
| tipchange.row | claude-image | 1 | 100.0% | 100.0% | 0 | 0.000 rows | 0.000 |  |
| tipchange.row | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.000 rows | 0.000 |  |

## Confusion matrices (rows = truth, columns = answer)

**tip.condition — claude-full**

| truth \ answer | blunt | double | good | unstable |
|---|---:|---:|---:|---:|
| blunt | 0 | 2 | 1 | 0 |
| double | 0 | 2 | 1 | 0 |
| unstable | 0 | 0 | 0 | 1 |

**tip.condition — claude-image**

| truth \ answer | blunt | double | good | unstable |
|---|---:|---:|---:|---:|
| blunt | 0 | 1 | 2 | 0 |
| double | 0 | 2 | 1 | 0 |
| unstable | 0 | 0 | 0 | 1 |

**tip.condition — baseline (classical)**

| truth \ answer | blunt | double | good | unstable |
|---|---:|---:|---:|---:|
| blunt | 3 | 0 | 0 | 0 |
| double | 0 | 1 | 2 | 0 |
| unstable | 0 | 0 | 0 | 1 |

