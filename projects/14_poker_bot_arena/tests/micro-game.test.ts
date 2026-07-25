import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  applyMicroAction,
  blueprintProbabilities,
  chooseMicroBotAction,
  createMicroGame,
  encodeMicroState,
  isMicroTerminal,
  legalMicroActions,
  microTerminalPayoffs,
  type BrowserStrategy,
  type MicroGame,
} from "../app/micro-game.ts";

test("micro Hold'em preserves the big blind option before revealing the flop", () => {
  let game = createMicroGame(0, () => 0.42);
  assert.deepEqual(game.contributions, [1, 2]);
  assert.deepEqual(legalMicroActions(game), [0, 1, 2, 3, 4]);
  game = applyMicroAction(game, 1);
  assert.equal(game.street, 0);
  assert.equal(game.currentPlayer, 1);
  assert.deepEqual(legalMicroActions(game), [1, 2, 3, 4]);
  game = applyMicroAction(game, 1);
  assert.equal(game.street, 1);
  assert.equal(game.currentPlayer, 1);
});

test("all-in and call run the board with zero-sum big-blind payoffs", () => {
  let game = createMicroGame(0, () => 0.17);
  game = applyMicroAction(game, 4);
  game = applyMicroAction(game, 1);
  assert.equal(isMicroTerminal(game), true);
  const payoffs = microTerminalPayoffs(game);
  assert.equal(payoffs[0] + payoffs[1], 0);
});

test("126-input observation hides opponent cards and the unrevealed board", () => {
  const game = createMicroGame(0, () => 0.31);
  const counterfactual: MicroGame = {
    ...game,
    holeCards: [game.holeCards[0], [0, 1]],
    board: [2, 3, 4],
  };
  assert.equal(encodeMicroState(game).length, 126);
  assert.deepEqual(encodeMicroState(game), encodeMicroState(counterfactual));
});

test("missing neural checkpoint falls back to a normalized legal policy", () => {
  const game = createMicroGame(0, () => 0.73);
  const probabilities = blueprintProbabilities(game, null);
  const total = legalMicroActions(game)
    .map((action) => probabilities[action])
    .reduce((sum, probability) => sum + probability, 0);
  assert.equal(total, 1);
});

test("browser features match the Python encoder on a fixed information state", () => {
  const template = createMicroGame(0, () => 0.51);
  const game: MicroGame = {
    ...template,
    holeCards: [[1, 14], [5, 22]],
    board: [3, 8, 17],
    button: 0,
    currentPlayer: 0,
    street: 0,
    contributions: [1, 2],
    streetContributions: [1, 2],
    acted: [false, false],
    raises: 0,
    history: [],
  };
  const features = encodeMicroState(game);
  const expected = new Map([
    [0, 1],
    [6, 1],
    [13, 1],
    [17, 1],
    [54, 1],
    [55, 0.075],
    [56, 0.05],
    [57, 0.1],
    [58, 0.95],
    [59, 0.9],
    [60, 0.05],
    [61, 0.1],
    [62, 0.05],
    [65, 1],
    [66, 1],
    [67, 1],
    [68, 1],
    [69, 1],
  ]);
  features.forEach((value, index) => {
    assert.ok(Math.abs(value - (expected.get(index) ?? 0)) < 1e-6);
  });
});

test("browser MLP matches Python checkpoint inference", async () => {
  const modelUrl = new URL("../public/micro-strategy.json", import.meta.url);
  const model = JSON.parse(await readFile(modelUrl, "utf8")) as BrowserStrategy;
  const template = createMicroGame(0, () => 0.51);
  const game: MicroGame = {
    ...template,
    holeCards: [[1, 14], [5, 22]],
    board: [3, 8, 17],
    button: 0,
    currentPlayer: 0,
    street: 0,
    contributions: [1, 2],
    streetContributions: [1, 2],
    acted: [false, false],
    raises: 0,
    history: [],
  };
  const probabilities = blueprintProbabilities(game, model);
  const pythonReference = [
    0.7139482309119988,
    0.21766171625865188,
    0.027749954963677104,
    0.013344006476038658,
    0.027296091389633608,
  ];
  probabilities.forEach((probability, action) => {
    assert.ok(Math.abs(probability - pythonReference[action]) < 1e-5);
  });
});

test("stratified range resolver returns a legal searched action", async () => {
  const modelUrl = new URL("../public/micro-strategy.json", import.meta.url);
  const model = JSON.parse(await readFile(modelUrl, "utf8")) as BrowserStrategy;
  const game = createMicroGame(0, () => 0.37);
  const decision = chooseMicroBotAction(game, "resolver", model, () => 0.41);
  assert.ok(legalMicroActions(game).includes(decision.action));
  assert.equal(decision.rangeSize, 231);
  assert.equal(decision.actionValues?.length, 5);
});
