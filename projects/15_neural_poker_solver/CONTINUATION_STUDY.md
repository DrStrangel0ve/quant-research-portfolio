# Checkpoint continuation and search study

This study asks a narrower question than the original Royal Micro Hold'em
experiment: can a saved Deep CFR blueprint be improved without selecting on
poker variance or silently weakening it elsewhere?

The answer from these seeded experiments is **not yet**. Several candidates
improved a targeted matchup or an equal-weight heuristic suite, but none passed
every predeclared promotion criterion. The original 1,600-iteration v1
checkpoint therefore remains the browser champion.

## Promotion protocol

Candidate and incumbent play identical deals and use identical action-sampling
streams wherever the policies permit it. A candidate is promoted only if:

1. the equal-weight improvement over uniform random, calling station, and
   pot-pressure has a positive 95% confidence lower bound;
2. no individual matchup has a negative 95% confidence upper bound; and
3. direct candidate-v1 duplicate cross-play has a 95% lower bound of at least
   `-0.10 BB/hand`.

Thresholds and mixture weights are selected on one seed and confirmed once on a
fresh seed. Failed candidates are retained as ablations, not folded into the
production model.

## Deep CFR continuation

Checkpoints now restore both advantage networks, the average-strategy network,
and the absolute CFR iteration. Because the checkpoint intentionally omits its
large replay reservoirs, replay-anchored continuation first samples the v1
behavioral policy into a fresh strategy reservoir.

| Candidate | Key change | Equal-weight suite delta | Direct vs v1 | Decision |
|---|---|---:|---:|---|
| v2 | +2,400 iterations, no replay anchor | -0.0326 `[-0.0899, +0.0247]` | -0.0771 `[-0.2220, +0.0679]` | Reject |
| v3 | 100k v1 anchor states, then +2,400 iterations | +0.0001 `[-0.0445, +0.0447]` | +0.0629 `[-0.0886, +0.2144]` | Reject |

The anchored v3 model revealed a useful specialization hidden by its flat suite
mean: it improved against pot pressure by `+0.1223 BB/hand`
`[+0.0347, +0.2098]`, but regressed against uniform random by `-0.1091`
`[-0.1889, -0.0293]`. It is used as an experimental pressure specialist, not
as the new blueprint.

## Mixture and public-history adaptation

Static v1/v3 mixtures at specialist weights `0.2`, `0.4`, `0.6`, and `0.8`
all had suite intervals crossing zero. The best point estimate was the `0.4`
mixture at `+0.0086 BB/hand` `[-0.0331, +0.0502]`.

A pressure-adaptive policy then used a transparent Bayesian classifier over
public opponent actions. It never observes the opponent's private cards. The
selected posterior threshold `0.50` passed its selection run:

- suite: `+0.0345 BB/hand` `[+0.0041, +0.0650]`;
- pressure: `+0.1113` `[+0.0262, +0.1963]`; and
- direct vs v1: `-0.0127` `[-0.2157, +0.1903]`.

The fresh 5,000-pair-per-opponent confirmation stayed positive but did not
exclude zero:

- suite: `+0.0139` `[-0.0065, +0.0343]`;
- pressure: `+0.0531` `[-0.0026, +0.1088]`;
- uniform random: `-0.0114` `[-0.0370, +0.0142]`; and
- direct vs v1 over 8,000 pairs: `+0.0566` `[-0.0670, +0.1802]`.

It was not promoted.

## Learned information-set values

The value model consumes the acting player's cards plus public state and
history. It has no feature path to hidden opponent cards. Three ablations made
the representation failure measurable:

| Model | Objective/features | Held-out RMSE | Target SD | R² |
|---|---|---:|---:|---:|
| value v1 | raw features, Huber loss | 6.424 | 6.377 | below zero |
| value v2 | standardized target, MSE | 6.383 | 6.419 | 0.011 |
| value v3 | v2 plus information-safe showdown equity | 6.261 | 6.439 | 0.055 |

Equity is exact on the visible flop and a deterministic, suit-canonical Monte
Carlo estimate preflop. The equity feature materially improved correlation
(`0.265` versus `0.211`) but the return target remains noisy. This is evidence
for training counterfactual values over repeated public-belief states rather
than simply scaling the current regression dataset.

## Batched flop search

The resolver now:

- replays all compatible opponent holdings through the policy in batches;
- samples the posterior range with a stratified estimator;
- expands immediate opponent responses with common random numbers;
- scores all non-terminal leaves in one value-network batch; and
- mixes the searched action conservatively with v1.

Restricting search to the flop removed the weakest value-model street and cut
the seeded smoke benchmark from `353.4` seconds to `81.6` seconds. A selected
15% search mixture then produced:

| Test | Result |
|---|---:|
| Equal-weight suite, 500 pairs/opponent | +0.0933 `[+0.0389, +0.1478]` |
| Uniform random | +0.0785 `[-0.0203, +0.1773]` |
| Calling station | +0.0825 `[-0.0028, +0.1678]` |
| Pot pressure | +0.1190 `[+0.0207, +0.2173]` |
| Direct vs v1, initial 2,000 pairs | +0.0541 `[-0.1965, +0.3047]` |
| Direct vs v1, fresh 8,000-pair safety run | -0.0235 `[-0.1462, +0.0992]` |

The fixed-opponent gain is real on this benchmark, but the high-power direct
lower bound misses the `-0.10` safety threshold. Smaller 5% and 10% search
mixtures reduced the point gain and still had suite intervals crossing zero.
No search policy was promoted.

## What improved

Even without a new champion, this iteration materially strengthens the
research project:

- resumable neural CFR with replay anchoring;
- paired candidate-minus-incumbent estimators;
- static mixtures and public-only Bayesian opponent adaptation;
- a reproducible information-set value-training pipeline;
- exact/seeded equity feature ablations;
- batched public-range inference and common-random-number search; and
- explicit selection, confirmation, and rejection artifacts.

The next credible step is to generate repeated counterfactual-value targets for
the same public belief states, validate them on an exactly enumerable small
game, and use a safe subgame gadget instead of an unconstrained root action
mixture.
