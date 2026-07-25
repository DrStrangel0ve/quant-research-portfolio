import assert from "node:assert/strict";
import test from "node:test";

import {
  applyAction,
  informationKey,
  legalActions,
  terminalPayoffs,
  type LeducGame,
} from "../app/game.ts";

function state(overrides: Partial<LeducGame> = {}): LeducGame {
  return {
    handId: 0,
    privateCards: [0, 4],
    publicCard: 1,
    smallBlind: 0,
    currentPlayer: 0,
    roundIndex: 0,
    contributions: [1, 2],
    raises: 0,
    nonRaiseActions: 0,
    foldedPlayer: null,
    showdown: false,
    history: [],
    events: [],
    ...overrides,
  };
}

test("small blind faces call, raise, or fold", () => {
  assert.deepEqual(legalActions(state()), [0, 1, 2]);
});

test("call-check reveals the public card and preserves first actor", () => {
  const revealed = applyAction(applyAction(state(), 0), 3);
  assert.equal(revealed.roundIndex, 1);
  assert.equal(revealed.currentPlayer, 0);
  assert.deepEqual(revealed.contributions, [2, 2]);
  assert.equal(
    informationKey(revealed, 0, true),
    "h0|b0|m2|o2|r1|t0,3",
  );
});

test("public pair wins at showdown and payoffs stay zero-sum", () => {
  let game = applyAction(applyAction(state(), 0), 3);
  game = applyAction(applyAction(game, 3), 3);
  const payoffs = terminalPayoffs(game);
  assert.ok(payoffs[0] > 0);
  assert.equal(payoffs[0] + payoffs[1], 0);
});

test("fold payoff is scaled to big blinds", () => {
  assert.deepEqual(terminalPayoffs(applyAction(state(), 2)), [-0.5, 0.5]);
});
