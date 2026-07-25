# Poker Lab: exact and neural arena

A deployable browser interface with two deliberately different audit levels:

- exact Leduc Hold'em policies from
  [project 13](../13_poker_cfr_lab/README.md), and
- Royal Micro Hold'em neural policies and range search from
  [project 15](../15_neural_poker_solver/README.md).

## Exact Leduc opponents

| Bot | Decision source |
|---|---|
| CFR+ 20K | 288-state perfect-recall average policy from the committed checkpoint |
| RLCard CFR | RLCard 1.2.0's bundled 84-state reference CFR checkpoint |
| Exact Punisher | Full-history pure best response to the CFR+ 20K policy |
| Pressure Bot | Raise-first deterministic heuristic |
| Calling Station | Call/check deterministic heuristic |
| Chaos Bot | Uniform legal random baseline |

The browser loads `public/poker-policies.json`; it does not call a model API or
pretend to recompute CFR live. Every probability shown in the policy trace comes
from the exported training table. Match score is device-local and intentionally
ephemeral.

## Neural no-limit opponents

| Bot | Decision source |
|---|---|
| Deep CFR | Browser MLP inference from the learned average-strategy checkpoint |
| Range Resolver | Bayesian range filter plus stratified common-random-number rollouts |
| Pot Pressure | Pot-first deterministic heuristic |
| Calling Station | Check/call deterministic heuristic |
| Chaos | Uniform legal random baseline |

The browser implements the same 126-feature suit-canonical encoder and three
dense layers as Python, then reads `public/micro-strategy.json`. No server or
model API sees the hand. Range Resolver resamples hidden cards from information
available to the bot; it never reads the player's actual cards.

## Run locally

```bash
npm install
npm run dev
npm test
```

Leduc shortcuts are `F` fold, `C` call, `K` check, and `R` raise. Micro no-limit
uses `F` fold, `C` check/call, `H` half-pot, `P` pot, and `A` all-in. The layout
also supports touch controls and reduced-motion preferences.

## Regenerate the policy bundle

From the repository root:

```bash
python projects/13_poker_cfr_lab/export_arena.py
```

Copy the resulting `projects/13_poker_cfr_lab/artifacts/arena_policies.json` to
`projects/14_poker_bot_arena/public/poker-policies.json`, then rebuild.

Regenerate the neural browser checkpoint with:

```bash
python projects/15_neural_poker_solver/run.py \
  --output projects/15_neural_poker_solver/artifacts \
  --browser-output projects/14_poker_bot_arena/public/micro-strategy.json
```
