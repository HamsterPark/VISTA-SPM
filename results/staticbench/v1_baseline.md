# STM-BabyVision results

Dataset: `tmp/staticbench/v1`

## Overall

| run | items | correct | accuracy | macro accuracy (subtypes) | mean score | abstained | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 230 | 213 | 92.6% | 93.1% | 0.948 | 0 | 0 | 0 |
| oracle | 230 | 230 | 100.0% | 100.0% | 1.000 | 0 | 0 | 0 |
| abstain | 230 | 0 | 0.0% | 0.0% | 0.000 | 230 | 0 | 0 |

## Per subtype

Accuracy counts an abstention or a missing answer as wrong; *answered* is accuracy over the items that got a valid answer. Errors are over answered items, in the unit shown (position lists: mean distance of matched points).

| subtype | run | n | accuracy | answered acc. | abstain | error median | error p90 | extra |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tip.condition | baseline | 50 | 92.0% | 92.0% | 0 | -  | - | recall blunt: 90.0%, dirty: 100.0%, double: 80.0%, good: 90.0%, unstable: 100.0% |
| tip.condition | oracle | 50 | 100.0% | 100.0% | 0 | -  | - | recall blunt: 100.0%, dirty: 100.0%, double: 100.0%, good: 100.0%, unstable: 100.0% |
| tip.condition | abstain | 50 | 0.0% | - | 50 | -  | - | recall blunt: 0.0%, dirty: 0.0%, double: 0.0%, good: 0.0%, unstable: 0.0% |
| adatoms.count | baseline | 20 | 80.0% | 80.0% | 0 | 0.000 atoms | 2.000 | within +-1: 85.0% |
| adatoms.count | oracle | 20 | 100.0% | 100.0% | 0 | 0.000 atoms | 0.000 | within +-1: 100.0% |
| adatoms.count | abstain | 20 | 0.0% | - | 20 | -  | - | within +-1: 0.0% |
| adatoms.positions | baseline | 20 | 65.0% | 65.0% | 0 | 0.037 nm | 0.075 | P 0.95 / R 0.92 / F1 0.91 |
| adatoms.positions | oracle | 20 | 100.0% | 100.0% | 0 | 0.000 nm | 0.000 | P 1.00 / R 1.00 / F1 1.00 |
| adatoms.positions | abstain | 20 | 0.0% | - | 20 | -  | - |  |
| move.detect | baseline | 20 | 100.0% | 100.0% | 0 | -  | - | recall False: 100.0%, True: 100.0% |
| move.detect | oracle | 20 | 100.0% | 100.0% | 0 | -  | - | recall False: 100.0%, True: 100.0% |
| move.detect | abstain | 20 | 0.0% | - | 20 | -  | - | recall False: 0.0%, True: 0.0% |
| move.distance | baseline | 10 | 100.0% | 100.0% | 0 | 0.017 nm | 0.030 |  |
| move.distance | oracle | 10 | 100.0% | 100.0% | 0 | 0.000 nm | 0.000 |  |
| move.distance | abstain | 10 | 0.0% | - | 10 | -  | - |  |
| move.which | baseline | 10 | 100.0% | 100.0% | 0 | 0.030 nm | 0.041 |  |
| move.which | oracle | 10 | 100.0% | 100.0% | 0 | 0.000 nm | 0.000 |  |
| move.which | abstain | 10 | 0.0% | - | 10 | -  | - |  |
| drift.vector | baseline | 20 | 100.0% | 100.0% | 0 | 0.032 nm | 0.064 |  |
| drift.vector | oracle | 20 | 100.0% | 100.0% | 0 | 0.000 nm | 0.000 |  |
| drift.vector | abstain | 20 | 0.0% | - | 20 | -  | - |  |
| herringbone.orientation | baseline | 20 | 100.0% | 100.0% | 0 | 0.142 deg | 0.353 |  |
| herringbone.orientation | oracle | 20 | 100.0% | 100.0% | 0 | 0.000 deg | 0.000 |  |
| herringbone.orientation | abstain | 20 | 0.0% | - | 20 | -  | - |  |
| herringbone.period | baseline | 20 | 100.0% | 100.0% | 0 | 0.004 nm | 0.035 |  |
| herringbone.period | oracle | 20 | 100.0% | 100.0% | 0 | 0.000 nm | 0.000 |  |
| herringbone.period | abstain | 20 | 0.0% | - | 20 | -  | - |  |
| step.height | baseline | 10 | 80.0% | 80.0% | 0 | 10.195 pm | 38.090 |  |
| step.height | oracle | 10 | 100.0% | 100.0% | 0 | 0.000 pm | 0.000 |  |
| step.height | abstain | 10 | 0.0% | - | 10 | -  | - |  |
| tipchange.row | baseline | 10 | 100.0% | 100.0% | 0 | 0.000 rows | 0.000 |  |
| tipchange.row | oracle | 10 | 100.0% | 100.0% | 0 | 0.000 rows | 0.000 |  |
| tipchange.row | abstain | 10 | 0.0% | - | 10 | -  | - |  |
| atomic.resolved | baseline | 20 | 100.0% | 100.0% | 0 | -  | - | recall False: 100.0%, True: 100.0% |
| atomic.resolved | oracle | 20 | 100.0% | 100.0% | 0 | -  | - | recall False: 100.0%, True: 100.0% |
| atomic.resolved | abstain | 20 | 0.0% | - | 20 | -  | - | recall False: 0.0%, True: 0.0% |

## Confusion matrices (rows = truth, columns = answer)

**tip.condition — baseline**

| truth \ answer | blunt | dirty | double | good | unstable |
|---|---:|---:|---:|---:|---:|
| blunt | 9 | 0 | 0 | 1 | 0 |
| dirty | 0 | 10 | 0 | 0 | 0 |
| double | 0 | 0 | 8 | 2 | 0 |
| good | 1 | 0 | 0 | 9 | 0 |
| unstable | 0 | 0 | 0 | 0 | 10 |

**tip.condition — oracle**

| truth \ answer | blunt | dirty | double | good | unstable |
|---|---:|---:|---:|---:|---:|
| blunt | 10 | 0 | 0 | 0 | 0 |
| dirty | 0 | 10 | 0 | 0 | 0 |
| double | 0 | 0 | 10 | 0 | 0 |
| good | 0 | 0 | 0 | 10 | 0 |
| unstable | 0 | 0 | 0 | 0 | 10 |

**tip.condition — abstain**

| truth \ answer | abstain | blunt | dirty | double | good | unstable |
|---|---:|---:|---:|---:|---:|---:|
| blunt | 10 | 0 | 0 | 0 | 0 | 0 |
| dirty | 10 | 0 | 0 | 0 | 0 | 0 |
| double | 10 | 0 | 0 | 0 | 0 | 0 |
| good | 10 | 0 | 0 | 0 | 0 | 0 |
| unstable | 10 | 0 | 0 | 0 | 0 | 0 |

**move.detect — baseline**

| truth \ answer | False | True |
|---|---:|---:|
| False | 10 | 0 |
| True | 0 | 10 |

**move.detect — oracle**

| truth \ answer | False | True |
|---|---:|---:|
| False | 10 | 0 |
| True | 0 | 10 |

**move.detect — abstain**

| truth \ answer | False | True | abstain |
|---|---:|---:|---:|
| False | 0 | 0 | 10 |
| True | 0 | 0 | 10 |

**atomic.resolved — baseline**

| truth \ answer | False | True |
|---|---:|---:|
| False | 10 | 0 |
| True | 0 | 10 |

**atomic.resolved — oracle**

| truth \ answer | False | True |
|---|---:|---:|
| False | 10 | 0 |
| True | 0 | 10 |

**atomic.resolved — abstain**

| truth \ answer | False | True | abstain |
|---|---:|---:|---:|
| False | 0 | 0 | 10 |
| True | 0 | 0 | 10 |

## By difficulty

| subtype | run | accuracy by difficulty |
|---|---|---|
| tip.condition | baseline | hard: 71.4% (n=7), normal: 95.3% (n=43) |
| adatoms.count | baseline | hard: 80.0% (n=10), normal: 80.0% (n=10) |
| adatoms.positions | baseline | hard: 70.0% (n=10), normal: 60.0% (n=10) |
| drift.vector | baseline | hard: 100.0% (n=1), normal: 100.0% (n=19) |
| herringbone.period | baseline | hard: 100.0% (n=8), normal: 100.0% (n=12) |
| tipchange.row | baseline | hard: 100.0% (n=1), normal: 100.0% (n=9) |
| tip.condition | oracle | hard: 100.0% (n=7), normal: 100.0% (n=43) |
| adatoms.count | oracle | hard: 100.0% (n=10), normal: 100.0% (n=10) |
| adatoms.positions | oracle | hard: 100.0% (n=10), normal: 100.0% (n=10) |
| drift.vector | oracle | hard: 100.0% (n=1), normal: 100.0% (n=19) |
| herringbone.period | oracle | hard: 100.0% (n=8), normal: 100.0% (n=12) |
| tipchange.row | oracle | hard: 100.0% (n=1), normal: 100.0% (n=9) |
| tip.condition | abstain | hard: 0.0% (n=7), normal: 0.0% (n=43) |
| adatoms.count | abstain | hard: 0.0% (n=10), normal: 0.0% (n=10) |
| adatoms.positions | abstain | hard: 0.0% (n=10), normal: 0.0% (n=10) |
| drift.vector | abstain | hard: 0.0% (n=1), normal: 0.0% (n=19) |
| herringbone.period | abstain | hard: 0.0% (n=8), normal: 0.0% (n=12) |
| tipchange.row | abstain | hard: 0.0% (n=1), normal: 0.0% (n=9) |

