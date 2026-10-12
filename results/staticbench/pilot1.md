# STM-BabyVision: Claude pilot

Dataset: `tmp\staticbench\v1`

## Overall

| run | items | correct | accuracy | macro accuracy (subtypes) | mean score | abstained | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-full | 12 | 12 | 100.0% | 100.0% | 1.000 | 0 | 0 | 0 |
| claude-image | 12 | 11 | 91.7% | 91.7% | 0.917 | 0 | 0 | 0 |
| baseline (classical) | 12 | 11 | 91.7% | 91.7% | 0.917 | 0 | 0 | 0 |

## Per subtype

Accuracy counts an abstention or a missing answer as wrong; *answered* is accuracy over the items that got a valid answer. Errors are over answered items, in the unit shown (position lists: mean distance of matched points).

| subtype | run | n | accuracy | answered acc. | abstain | error median | error p90 | extra |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tip.condition | claude-full | 1 | 100.0% | 100.0% | 0 | -  | - | recall unstable: 100.0% |
| tip.condition | claude-image | 1 | 100.0% | 100.0% | 0 | -  | - | recall unstable: 100.0% |
| tip.condition | baseline (classical) | 1 | 100.0% | 100.0% | 0 | -  | - | recall unstable: 100.0% |
| adatoms.count | claude-full | 1 | 100.0% | 100.0% | 0 | 0.000 atoms | 0.000 | within +-1: 100.0% |
| adatoms.count | claude-image | 1 | 100.0% | 100.0% | 0 | 0.000 atoms | 0.000 | within +-1: 100.0% |
| adatoms.count | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.000 atoms | 0.000 | within +-1: 100.0% |
| adatoms.positions | claude-full | 1 | 100.0% | 100.0% | 0 | 0.059 nm | 0.059 | P 1.00 / R 1.00 / F1 1.00 |
| adatoms.positions | claude-image | 1 | 100.0% | 100.0% | 0 | 0.026 nm | 0.026 | P 1.00 / R 1.00 / F1 1.00 |
| adatoms.positions | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.034 nm | 0.034 | P 1.00 / R 1.00 / F1 1.00 |
| move.detect | claude-full | 1 | 100.0% | 100.0% | 0 | -  | - | recall True: 100.0% |
| move.detect | claude-image | 1 | 100.0% | 100.0% | 0 | -  | - | recall True: 100.0% |
| move.detect | baseline (classical) | 1 | 100.0% | 100.0% | 0 | -  | - | recall True: 100.0% |
| move.distance | claude-full | 1 | 100.0% | 100.0% | 0 | 0.006 nm | 0.006 |  |
| move.distance | claude-image | 1 | 100.0% | 100.0% | 0 | 0.054 nm | 0.054 |  |
| move.distance | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.015 nm | 0.015 |  |
| move.which | claude-full | 1 | 100.0% | 100.0% | 0 | 0.023 nm | 0.023 |  |
| move.which | claude-image | 1 | 100.0% | 100.0% | 0 | 0.027 nm | 0.027 |  |
| move.which | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.021 nm | 0.021 |  |
| drift.vector | claude-full | 1 | 100.0% | 100.0% | 0 | 0.002 nm | 0.002 |  |
| drift.vector | claude-image | 1 | 100.0% | 100.0% | 0 | 0.014 nm | 0.014 |  |
| drift.vector | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.009 nm | 0.009 |  |
| herringbone.orientation | claude-full | 1 | 100.0% | 100.0% | 0 | 0.064 deg | 0.064 |  |
| herringbone.orientation | claude-image | 1 | 100.0% | 100.0% | 0 | 0.936 deg | 0.936 |  |
| herringbone.orientation | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.076 deg | 0.076 |  |
| herringbone.period | claude-full | 1 | 100.0% | 100.0% | 0 | 0.012 nm | 0.012 |  |
| herringbone.period | claude-image | 1 | 0.0% | 0.0% | 0 | 0.638 nm | 0.638 |  |
| herringbone.period | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.006 nm | 0.006 |  |
| step.height | claude-full | 1 | 100.0% | 100.0% | 0 | 3.400 pm | 3.400 |  |
| step.height | claude-image | 1 | 100.0% | 100.0% | 0 | 0.600 pm | 0.600 |  |
| step.height | baseline (classical) | 1 | 0.0% | 0.0% | 0 | 38.250 pm | 38.250 |  |
| tipchange.row | claude-full | 1 | 100.0% | 100.0% | 0 | 0.000 rows | 0.000 |  |
| tipchange.row | claude-image | 1 | 100.0% | 100.0% | 0 | 0.000 rows | 0.000 |  |
| tipchange.row | baseline (classical) | 1 | 100.0% | 100.0% | 0 | 0.000 rows | 0.000 |  |
| atomic.resolved | claude-full | 1 | 100.0% | 100.0% | 0 | -  | - | recall True: 100.0% |
| atomic.resolved | claude-image | 1 | 100.0% | 100.0% | 0 | -  | - | recall True: 100.0% |
| atomic.resolved | baseline (classical) | 1 | 100.0% | 100.0% | 0 | -  | - | recall True: 100.0% |

## Confusion matrices (rows = truth, columns = answer)

**tip.condition — claude-full**

| truth \ answer | unstable |
|---|---:|
| unstable | 1 |

**tip.condition — claude-image**

| truth \ answer | unstable |
|---|---:|
| unstable | 1 |

**tip.condition — baseline (classical)**

| truth \ answer | unstable |
|---|---:|
| unstable | 1 |

**move.detect — claude-full**

| truth \ answer | True |
|---|---:|
| True | 1 |

**move.detect — claude-image**

| truth \ answer | True |
|---|---:|
| True | 1 |

**move.detect — baseline (classical)**

| truth \ answer | True |
|---|---:|
| True | 1 |

**atomic.resolved — claude-full**

| truth \ answer | True |
|---|---:|
| True | 1 |

**atomic.resolved — claude-image**

| truth \ answer | True |
|---|---:|
| True | 1 |

**atomic.resolved — baseline (classical)**

| truth \ answer | True |
|---|---:|
| True | 1 |

