# Poker Lab arena

A deployable browser interface for the independently trained Leduc Hold'em
policies in [project 13](../13_poker_cfr_lab/README.md).

## Opponents

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

## Run locally

```bash
npm install
npm run dev
npm test
```

Keyboard shortcuts are `F` fold, `C` call, `K` check, and `R` raise. The layout
also supports touch controls and reduced-motion preferences.

## Regenerate the policy bundle

From the repository root:

```bash
python projects/13_poker_cfr_lab/export_arena.py
```

Copy the resulting `projects/13_poker_cfr_lab/artifacts/arena_policies.json` to
`projects/14_poker_bot_arena/public/poker-policies.json`, then rebuild.
