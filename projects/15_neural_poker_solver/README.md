# Neural poker solver: scaling beyond a tabular game

Project 13 answers a narrow question exactly: can a from-scratch CFR+ policy
approach equilibrium in six-card Leduc? This project asks the next engineering
question: how do the representation, sampling, function approximation, and
decision-time search change when the tree no longer fits in a table?

The benchmark is **Royal Micro Hold'em**, a synthetic heads-up no-limit game
with 24 cards (`9` through `A`, four suits), two private cards, a three-card
flop, 20-chip stacks, two betting streets, and the action abstraction
`{fold, check/call, half-pot, pot, all-in}`. It is materially larger than Leduc
and supports real five-card hand evaluation, range inference, bet sizing, and
all-ins. It is still far smaller than full heads-up no-limit Texas Hold'em.

## What is implemented

- Immutable no-limit engine with heads-up action order, effective stacks,
  blind option, bounded raises, all-in runouts, and zero-sum BB utilities.
- Complete five-card evaluator covering high card through straight flush.
- Exact suit-isomorphism canonicalization across all 24 suit permutations.
- A 126-dimensional information-state tensor containing only observable cards,
  position, stacks, pot, price, legal actions, street, and public action history.
- External-sampling **Deep CFR** from scratch in PyTorch:
  - one advantage MLP per player,
  - positive-regret matching during traversals,
  - bounded reservoir replay,
  - iteration-weighted advantage and average-strategy samples, and
  - a separately trained average-policy network.
- Dependency-free JSON export of the trained MLP for the browser arena.
- Seat-swapped duplicate-poker evaluation with paired 95% confidence intervals.
- A Bayesian public-range filter and lightweight rollout resolver that searches
  each legal root action against the learned blueprint.
- Checkpoint continuation with replay anchoring and paired promotion gates.
- Public-history pressure adaptation, static checkpoint mixtures, and
  candidate-minus-incumbent duplicate evaluation.
- An information-set value network with exact/seeded showdown-equity features,
  batched range inference, and conservative flop-only value search.
- Tests for rules, accounting, evaluator order, suit invariance, hidden-card
  isolation, neural masks, replay, duplicate evaluation, and belief support.

The implementation draws on the scaling ideas in
[Deep CFR](https://arxiv.org/abs/1811.00164), while the resolver is a deliberately
smaller stepping stone toward public-belief methods such as
[ReBeL](https://arxiv.org/abs/2007.13544). It does **not** claim ReBeL's safe
subgame-solving guarantees.

## Reproduce

Install the optional neural dependency, train, benchmark, and export the browser
policy:

```bash
python -m pip install -e ".[dev,neural-poker]"
python projects/15_neural_poker_solver/run.py \
  --iterations 1600 --traversals 6 --duplicate-pairs 2000 \
  --browser-output projects/14_poker_bot_arena/public/micro-strategy.json
```

The run writes:

- `deep_cfr_checkpoint.pt`: PyTorch weights plus exact training configuration.
- `micro_strategy.json`: small MLP weights in a browser-readable format.
- `training_diagnostics.csv` and `.png`: loss and replay growth by checkpoint.
- `summary.json`: duplicate cross-play, confidence intervals, and resolver audit.

Continue from that checkpoint without discarding the learned networks:

```bash
python projects/15_neural_poker_solver/continue_training.py \
  --additional-iterations 2400 --traversals 8 --anchor-samples 100000 \
  --promote-browser projects/14_poker_bot_arena/public/micro-strategy.json
```

Continuation retains both advantage networks, the average-strategy network, and
the absolute CFR iteration counter. Because checkpoints omit large historical
reservoirs, it first reconstructs average-policy replay by sampling the saved
blueprint. This anchor prevents end-of-run strategy fitting from catastrophically
forgetting the source policy. Advantage replay starts fresh. Promotion is
automatic only when a predeclared paired gate passes:
the equal-weight fixed-opponent improvement interval must exclude zero, no
individual opponent may show a significant regression, and direct v2-vs-v1
cross-play may not have a lower confidence bound below `-0.10 BB/hand`.

Train the information-set value model and evaluate conservative flop search:

```bash
python projects/15_neural_poker_solver/train_value_model.py \
  --examples 30000 --rollout-repeats 2 --epochs 25 \
  --output projects/15_neural_poker_solver/artifacts/value_v3_equity
python projects/15_neural_poker_solver/evaluate_value_search.py \
  --value-checkpoint \
    projects/15_neural_poker_solver/artifacts/value_v3_equity/state_value_checkpoint.pt \
  --weights 0.05 0.10 0.15 --belief-samples 32 --search-streets 1
```

See [CONTINUATION_STUDY.md](CONTINUATION_STUDY.md) for the full continuation,
mixture, adaptation, value-model, and search ablations. No candidate cleared
every predeclared gate, so the committed v1 browser checkpoint remains the
production champion.

## Verified seeded result

The committed checkpoint used seed 15,001, 1,600 iterations, six traversals per
player per iteration, and CPU training. It collected 43,438/45,071 advantage
samples and filled a 100,000-sample average-strategy reservoir in 114.6 seconds.

| Duplicate matchup | Mean BB/hand | Paired 95% CI |
|---|---:|---:|
| Neural blueprint vs uniform random | +0.613 | [+0.393, +0.834] |
| Neural blueprint vs calling station | +0.367 | [+0.081, +0.653] |
| Neural blueprint vs pot pressure | +0.262 | [-0.067, +0.591] |
| Neural blueprint self-play control | +0.026 | [-0.220, +0.272] |
| Belief rollout resolver vs blueprint | -0.075 | [-1.572, +1.422] |

The first two edges exclude zero over 2,000 duplicate pairs. Pot pressure is
unresolved. The 50-pair resolver comparison is intentionally reported as
inconclusive; its wide interval is evidence that the decision-time search and
evaluation budget are the next bottlenecks, not evidence of parity.

## What the benchmark can and cannot establish

Duplicate matches substantially reduce card and position variance, but winning
against random or heuristic bots is not an equilibrium certificate. Exact
best-response traversal is intentionally retained in Project 13 as a small-game
oracle. For this larger game, the rollout resolver is reported as an
**approximate local response**, not exploitability.

The next frontier step would replace the one-step rollout search with a
depth-limited public-belief subgame solver, train on much larger distributed
traversal budgets, add learned counterfactual values at the depth boundary, and
evaluate with low-variance protocols against external HUNL agents.

See the [frontier gap and engineering roadmap](SOTA_GAP.md) for the capability
matrix and milestone-level acceptance criteria.

## Interview prompts

1. Why is the average-strategy network separate from the advantage networks?
2. Where does external sampling introduce variance, and what remains unbiased?
3. Why must suit canonicalization preserve suit-sharing relationships between
   private cards and the board?
4. Why is a rollout response not the same as a safe continual resolver?
5. What evidence would be required before comparing this system with Slumbot,
   GTO Wizard, Libratus, or ReBeL?
