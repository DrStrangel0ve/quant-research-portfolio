# Poker CFR+ lab: imperfect-information optimization from scratch

This project trains a poker agent without a poker-learning framework, audits it
with exact game-tree calculations, and seats it against RLCard's published
pretrained CFR checkpoint. The benchmark is heads-up **Leduc Hold'em**, a small
research game that retains the hard parts of poker: hidden private information,
a public reveal, bluffing, sequential betting, and imperfect information.

The narrow game is intentional. Its complete chance space has only 240 ordered
deals/blind assignments, so claims can be checked with exact expectation and
best-response calculations rather than a flattering sample win rate. This is a
game-solving project, not a claim to have reproduced a large-scale no-limit
system such as Pluribus.

## What is implemented

- Immutable six-card Leduc engine with RLCard-compatible blinds, betting
  amounts, raise cap, round transitions, showdown, and big-blind-scaled payoffs.
- Chance-sampled **CFR+** written from first principles: alternating traversals,
  counterfactual reach, positive regret matching, regret flooring, and linearly
  weighted average strategies.
- Two information models: RLCard's compact 84-state observation abstraction and
  a stronger 288-state perfect-recall public-history model.
- Exact enumeration of self-play value and every policy matchup.
- Exact full-history best response and exploitability (half of NashConv).
- Duplicate-poker Monte Carlo with identical card/blind outcomes seat-swapped,
  paired standard errors, and 95% confidence intervals.
- Transparent random, calling-station, and always-aggressive baselines.
- Optional adapter for RLCard 1.2.0's bundled `leduc-holdem-cfr` checkpoint.
- JSON checkpoint export used by the browser arena in project 14.

## Game and utility

The deck is `J♠ J♥ Q♠ Q♥ K♠ K♥`. Each player receives one private card.
The small and big blinds contribute one and two chips. After the first betting
round, one public card is revealed; fixed raises cost two chips before the reveal
and four after it, with at most two raises per round. A private/public rank match
is a pair; otherwise the higher private rank wins. Net chip payoffs are divided
by the two-chip big blind.

Every terminal node is tested for zero-sum accounting. The engine is also driven
through seeded random action sequences beside RLCard, asserting identical legal
actions, contributions, acting player, transitions, and terminal payoffs.

## CFR+ update

For information state \(I\), legal action \(a\), and iteration \(t\):

\[
R_t^+(I,a) = \max\left(0,\ R_{t-1}^+(I,a)
  + \pi_{-i}(I)\left[v_i(I,a)-v_i(I)\right]\right).
\]

The current strategy normalizes positive regrets, falling back to uniform legal
play when all regrets are zero. The reported policy is not the last iterate. It
is the linearly weighted behavioral average
\(\sum_t t\,\pi_i(I)\,\sigma_t(I)\), normalized per information state.
Chance sampling chooses a blind assignment and three distinct cards, while each
player update still traverses every legal betting action.

## Reproduce

```bash
python -m pip install -e ".[dev,poker]"
python projects/13_poker_cfr_lab/run.py --iterations 3000 --duplicate-pairs 2000
```

Outputs:

- `results/cfr_plus_checkpoint.json`: policy, regrets, training metadata.
- `results/convergence.csv`: exact exploitability and random-bot value by
  training checkpoint.
- `results/summary.json`: exact baseline table and the RLCard face-off with a
  paired confidence interval.

The committed browser policy was trained longer with:

```bash
python projects/13_poker_cfr_lab/run.py \
  --iterations 20000 --duplicate-pairs 10000 \
  --output projects/13_poker_cfr_lab/artifacts
```

## Verified 20,000-iteration result

| Audit | Result |
|---|---:|
| Exact exploitability | 0.04236 BB/hand |
| Exact value vs uniform random | +0.73821 BB/hand |
| Exact value vs calling station | +0.30810 BB/hand |
| Exact value vs always aggressive | +0.41667 BB/hand |
| Exact value vs RLCard CFR, either seat | +0.13596 BB/hand |
| 10,000-pair duplicate result vs RLCard | +0.10673 BB/hand |
| Duplicate 95% confidence interval | [+0.06749, +0.14596] BB/hand |

Exact exploitability fell from 0.62796 after 100 iterations to 0.19166 after
1,000 and 0.04236 after 20,000. These are results for the six-card Leduc game,
not transferable win-rate claims for Texas Hold'em.

## Evaluation discipline

Head-to-head profit alone is not evidence of equilibrium quality: one policy
can exploit a specific opponent while remaining easy for a third policy to
exploit. This project therefore reports both cross-play and an adversarial
best-response score. The best responder sees the complete public action history,
even when it attacks RLCard's compressed observation policy. That choice makes
the audit stricter.

RLCard is used only to load the external reference checkpoint. The rules engine,
trainer, exact evaluator, baselines, saved strategy, and interactive arena remain
independent. OpenSpiel's CFR/CFR+ and exploitability implementations are useful
research references, but its default Leduc rules differ in blind/ante and action
order, so mixing those scores into this table would not be an apples-to-apples
match.

## Limitations

- Leduc is tiny compared with heads-up no-limit Texas Hold'em.
- CFR+ convergence in this stochastic trainer depends on the seed and iteration
  budget; the convergence table makes that path visible.
- The RLCard checkpoint is a reproducible library baseline, not a claim about
  the current frontier of large-scale poker AI.
- The arena exposes probabilities for learning, so it is not designed as a
  hidden-policy competitive product.

## Interview prompts

1. Why does counterfactual reach exclude the updating player's own action reach?
2. Why can a bot beat one reference while having worse exploitability?
3. What bias is introduced by merging public histories into an 84-state
   observation abstraction?
4. How would external-sampling MCCFR, Deep CFR, or subgame solving change the
   scaling story for larger poker games?
