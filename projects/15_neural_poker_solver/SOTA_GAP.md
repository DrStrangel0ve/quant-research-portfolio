# Frontier gap and engineering roadmap

Royal Micro Hold'em is a useful scaling experiment, not a state-of-the-art poker
agent. This note makes the distance concrete and turns “make it stronger” into
testable engineering milestones.

## Capability matrix

| Dimension | This project | Frontier direction |
|---|---|---|
| Game | 24 cards, heads-up, two streets, five abstract actions | 52-card HUNL or multiplayer NLHE with all streets and richer sizing |
| Blueprint | Two 128-unit advantage MLPs plus one average-policy MLP | Much larger distributed self-play, stronger architectures, and more data |
| State | Suit-canonical private/public cards, stacks, price, and history | Explicit public belief state with counterfactual values over both ranges |
| Search | One-step action choice with posterior-weighted blueprint rollouts | Depth-limited or nested subgame solving with safety constraints |
| Compute | 114.6 CPU seconds; 9,600 sampled traversals | Large parallel CPU/GPU clusters and orders of magnitude more trajectories |
| Equilibrium audit | Exact only in Project 13 Leduc; approximate here | Local best response, low-variance evaluation, and external agent cross-play |
| Deployment | Dependency-free browser inference and local range search | Optimized batched inference/search with strict latency budgets |

The relevant research lineage is:

- [Deep CFR](https://arxiv.org/abs/1811.00164): neural approximation of
  counterfactual advantages and average strategy.
- [Discounted CFR](https://arxiv.org/abs/1809.04040): iteration discounting that
  can accelerate practical regret minimization.
- [Libratus](https://doi.org/10.1126/science.aao1733): blueprint abstraction plus
  nested subgame solving in heads-up no-limit poker.
- [ReBeL](https://arxiv.org/abs/2007.13544): self-play reinforcement learning
  and search over public belief states.
- [Pluribus](https://doi.org/10.1126/science.aay2400): blueprint plus real-time
  search for multiplayer no-limit poker.
- [Student of Games](https://arxiv.org/abs/2112.03178): a general search and
  learning framework spanning perfect- and imperfect-information games.
- [GTO Wizard AI benchmark (2026)](https://arxiv.org/abs/2603.23660): a recent
  example of external Slumbot cross-play and AIVAT-style variance reduction.

## Measured current boundary

The 1,600-iteration checkpoint has statistically positive duplicate-poker edges
against uniform random and a calling station. Its interval against a pot-first
pressure bot includes zero. A 32-rollout belief response evaluated over only 50
duplicate pairs is also unresolved.

Those outcomes identify two different bottlenecks:

1. **Blueprint quality:** the network needs more and better counterfactual data,
   stable discounting, and convergence diagnostics beyond replay loss.
2. **Search quality:** sampled terminal returns are too noisy to serve as strong
   leaf values, and root-only action search cannot protect the blueprint across
   a complete subgame.

## Milestones with acceptance criteria

### 1. Neural small-game oracle

Implement the same neural pipeline on Leduc and require:

- exact exploitability below `0.05 BB/hand`,
- three seeds with reported median and range,
- policy/value regression against tabular CFR+, and
- ablations for replay capacity, network width, and traversal count.

This separates function-approximation bugs from large-game approximation.

### 2. Stronger blueprint

Add linear CFR/DCFR weighting, batched traversals, and opponent-reach features.
Acceptance criteria:

- monotone improvement in a held-out local-best-response proxy,
- positive duplicate lower confidence bounds against all fixed heuristics,
- stable results across at least three seeds, and
- throughput and memory scaling curves.

### 3. Learned public-belief values

Represent both players' conditional ranges and train a value network on public
states. Acceptance criteria:

- calibrated counterfactual-value error on held-out subgames,
- exact agreement on enumerable turn/river micro-subgames, and
- no feature path containing hidden opponent cards.

### 4. Safe depth-limited resolving

Replace root rollouts with a subgame gadget and depth-limited CFR search.
Acceptance criteria:

- exploitability does not increase on exact small-game tests,
- search improves a local-best-response score over the blueprint,
- action quality improves as search iterations increase, and
- p95 decision latency is reported.

### 5. Full-deck benchmark

Move to 52 cards, add turn and river, and expand bet sizing. Use card
canonicalization, range tensors, batched GPU traversal, and checkpointed
distributed workers. Do not claim frontier strength until the agent has:

- reproducible duplicate cross-play against a credible external HUNL baseline,
- a low-variance estimator such as a correctly implemented AIVAT protocol,
- local-best-response or restricted NashConv evidence, and
- disclosed hardware, training samples, wall time, seeds, and confidence
  intervals.

## Recruiting signal

The strongest portfolio story is not “I built a SOTA poker bot.” It is:

> I moved from an exactly auditable tabular equilibrium solver to a larger
> neural imperfect-information system, preserved information-set correctness
> across Python and browser runtimes, measured where approximation became
> inconclusive, and specified the experiments required to close each gap.
