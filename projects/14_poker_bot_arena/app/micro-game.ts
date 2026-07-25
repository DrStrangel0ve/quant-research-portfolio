export type MicroAction = 0 | 1 | 2 | 3 | 4;
export type MicroPlayer = 0 | 1;

export const MICRO_ACTION_NAMES: Record<MicroAction, string> = {
  0: "Fold",
  1: "Check / Call",
  2: "½ Pot",
  3: "Pot",
  4: "All-in",
};

const STARTING_STACK = 20;
const ACTION_COUNT = 5;
const FEATURE_DIM = 126;
const LEGAL_MASK_OFFSET = 65;
const HISTORY_OFFSET = 70;
const MAX_HISTORY = 8;

export interface MicroHistory {
  street: 0 | 1;
  action: MicroAction;
}

export interface MicroGame {
  handId: number;
  holeCards: [[number, number], [number, number]];
  board: [number, number, number];
  button: MicroPlayer;
  currentPlayer: MicroPlayer;
  street: 0 | 1;
  contributions: [number, number];
  streetContributions: [number, number];
  acted: [boolean, boolean];
  raises: number;
  foldedPlayer: MicroPlayer | null;
  showdown: boolean;
  history: MicroHistory[];
  events: string[];
}

export interface BrowserStrategy {
  format: "quantlab-royal-micro-strategy-v1";
  training: {
    iterations: number;
    completed_iterations: number;
    traversals_per_player: number;
  };
  network: {
    input_size: number;
    hidden_size: number;
    activation: "relu";
    weights: Array<{ weight: number[][]; bias: number[] }>;
  };
}

export interface MicroDecision {
  action: MicroAction;
  probabilities: number[];
  key: string;
  actionValues?: number[];
  rangeSize?: number;
}

export type MicroBotId =
  | "blueprint"
  | "resolver"
  | "pressure"
  | "calling"
  | "random";

export function createMicroGame(
  handId: number,
  random = Math.random,
): MicroGame {
  const deck = Array.from({ length: 24 }, (_, index) => index);
  shuffle(deck, random);
  const button = (handId % 2) as MicroPlayer;
  return initialMicroGame(
    handId,
    [
      [deck[0], deck[1]],
      [deck[2], deck[3]],
    ],
    [deck[4], deck[5], deck[6]],
    button,
  );
}

export function isMicroTerminal(game: MicroGame): boolean {
  return game.foldedPlayer !== null || game.showdown;
}

export function microStacks(game: MicroGame): [number, number] {
  return [
    STARTING_STACK - game.contributions[0],
    STARTING_STACK - game.contributions[1],
  ];
}

export function microToCall(
  game: MicroGame,
  player = game.currentPlayer,
): number {
  return Math.max(...game.streetContributions) - game.streetContributions[player];
}

export function legalMicroActions(game: MicroGame): MicroAction[] {
  if (isMicroTerminal(game)) return [];
  const player = game.currentPlayer;
  const opponent = (1 - player) as MicroPlayer;
  const stacks = microStacks(game);
  const outstanding = microToCall(game, player);
  const actions: MicroAction[] = outstanding > 0 ? [0, 1] : [1];
  const canRaise =
    stacks[player] > outstanding && stacks[opponent] > 0 && game.raises < 2;
  if (!canRaise) return actions;

  const seen = new Set<number>();
  for (const action of [2, 3] as MicroAction[]) {
    const chips = microChipsFor(game, action);
    if (chips <= outstanding || chips >= stacks[player] || seen.has(chips)) continue;
    actions.push(action);
    seen.add(chips);
  }
  if (stacks[player] > outstanding) actions.push(4);
  return actions;
}

export function applyMicroAction(
  game: MicroGame,
  action: MicroAction,
): MicroGame {
  if (!legalMicroActions(game).includes(action)) {
    throw new Error(`${MICRO_ACTION_NAMES[action]} is not legal`);
  }
  const player = game.currentPlayer;
  const opponent = (1 - player) as MicroPlayer;
  const history = [...game.history, { street: game.street, action }];
  const actor = player === 0 ? "You" : "Bot";
  const events = [
    ...game.events,
    `${game.street === 0 ? "Preflop" : "Flop"} · ${actor} ${MICRO_ACTION_NAMES[action].toLowerCase()}`,
  ];
  if (action === 0) {
    return {
      ...game,
      currentPlayer: opponent,
      foldedPlayer: player,
      history,
      events,
    };
  }

  const contributions: [number, number] = [...game.contributions];
  const streetContributions: [number, number] = [...game.streetContributions];
  const chips = microChipsFor(game, action);
  contributions[player] += chips;
  streetContributions[player] += chips;
  let acted: [boolean, boolean] = [...game.acted];
  let raises = game.raises;
  if (action >= 2) {
    acted = [false, false];
    acted[player] = true;
    raises += 1;
  } else {
    acted[player] = true;
  }
  const advanced: MicroGame = {
    ...game,
    currentPlayer: opponent,
    contributions,
    streetContributions,
    acted,
    raises,
    history,
    events,
  };
  const balanced = streetContributions[0] === streetContributions[1];
  if (!(balanced && acted[0] && acted[1])) return advanced;
  const stacks = microStacks(advanced);
  if (stacks.includes(0) || game.street === 1) {
    return {
      ...advanced,
      showdown: true,
      events: [...events, "Showdown · board run complete"],
    };
  }
  return {
    ...advanced,
    currentPlayer: (1 - game.button) as MicroPlayer,
    street: 1,
    streetContributions: [0, 0],
    acted: [false, false],
    raises: 0,
    events: [
      ...events,
      `Flop · ${game.board.map(microCardLabel).join(" ")} revealed`,
    ],
  };
}

export function microTerminalPayoffs(game: MicroGame): [number, number] {
  if (!isMicroTerminal(game)) throw new Error("The hand is not over");
  const pot = game.contributions[0] + game.contributions[1];
  let winners: MicroPlayer[];
  if (game.foldedPlayer !== null) {
    winners = [(1 - game.foldedPlayer) as MicroPlayer];
  } else {
    const scores = ([0, 1] as MicroPlayer[]).map((player) =>
      evaluateFive([...game.holeCards[player], ...game.board]),
    );
    const comparison = compareTuple(scores[0], scores[1]);
    winners = comparison === 0 ? [0, 1] : [comparison > 0 ? 0 : 1];
  }
  const shares: [number, number] = [0, 0];
  for (const winner of winners) shares[winner] = pot / winners.length;
  return [
    (shares[0] - game.contributions[0]) / 2,
    (shares[1] - game.contributions[1]) / 2,
  ];
}

export function microCardLabel(card: number): string {
  const ranks = ["9", "T", "J", "Q", "K", "A"];
  const suits = ["♣", "♦", "♥", "♠"];
  return `${ranks[Math.floor(card / 4)]}${suits[card % 4]}`;
}

export function encodeMicroState(
  game: MicroGame,
  player = game.currentPlayer,
): number[] {
  const features = Array.from({ length: FEATURE_DIM }, () => 0);
  const board = game.street === 1 || game.showdown ? game.board : [];
  const [hole, canonicalBoard] = canonicalCards(game.holeCards[player], board);
  [...hole, ...canonicalBoard].forEach((card, slot) => {
    const start = slot * 10;
    features[start + Math.floor(card / 4)] = 1;
    features[start + 6 + (card % 4)] = 1;
  });
  canonicalBoard.forEach((_, index) => {
    features[50 + index] = 1;
  });
  const opponent = (1 - player) as MicroPlayer;
  const stacks = microStacks(game);
  const scalars = [
    game.street,
    Number(player === game.button),
    (game.contributions[0] + game.contributions[1]) / 40,
    game.contributions[player] / 20,
    game.contributions[opponent] / 20,
    stacks[player] / 20,
    stacks[opponent] / 20,
    game.streetContributions[player] / 20,
    game.streetContributions[opponent] / 20,
    microToCall(game, player) / 20,
    game.raises / 2,
    game.history.length / MAX_HISTORY,
  ];
  scalars.forEach((value, index) => {
    features[53 + index] = value;
  });
  for (const action of legalMicroActions(game)) {
    features[LEGAL_MASK_OFFSET + action] = 1;
  }
  game.history.slice(-MAX_HISTORY).forEach((entry, index) => {
    const start = HISTORY_OFFSET + index * 7;
    features[start + entry.action] = 1;
    features[start + ACTION_COUNT + entry.street] = 1;
  });
  return features;
}

export function blueprintProbabilities(
  game: MicroGame,
  model: BrowserStrategy | null,
): number[] {
  const legal = legalMicroActions(game);
  if (!model || model.network.input_size !== FEATURE_DIM) {
    return uniformProbabilities(legal);
  }
  let activations = encodeMicroState(game);
  model.network.weights.forEach((layer, layerIndex) => {
    const output = layer.weight.map(
      (row, unit) =>
        row.reduce((sum, weight, index) => sum + weight * activations[index], 0) +
        layer.bias[unit],
    );
    activations =
      layerIndex === model.network.weights.length - 1
        ? output
        : output.map((value) => Math.max(0, value));
  });
  const maximum = Math.max(...legal.map((action) => activations[action]));
  const exponentials = legal.map((action) =>
    Math.exp(activations[action] - maximum),
  );
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  const probabilities = Array.from({ length: ACTION_COUNT }, () => 0);
  legal.forEach((action, index) => {
    probabilities[action] = exponentials[index] / total;
  });
  return probabilities;
}

export function chooseMicroBotAction(
  game: MicroGame,
  bot: MicroBotId,
  model: BrowserStrategy | null,
  random = Math.random,
): MicroDecision {
  const legal = legalMicroActions(game);
  let probabilities = Array.from({ length: ACTION_COUNT }, () => 0);
  let actionValues: number[] | undefined;
  let rangeSize: number | undefined;
  if (bot === "blueprint") {
    probabilities = blueprintProbabilities(game, model);
  } else if (bot === "calling") {
    probabilities[1] = 1;
  } else if (bot === "pressure") {
    const choice = ([3, 2, 4, 1, 0] as MicroAction[]).find((action) =>
      legal.includes(action),
    );
    probabilities[choice ?? legal[0]] = 1;
  } else if (bot === "resolver" && model) {
    const result = resolveAction(game, model, 14, random);
    probabilities[result.action] = 1;
    actionValues = result.values;
    rangeSize = result.rangeSize;
  } else {
    probabilities = uniformProbabilities(legal);
  }
  const action = sampleAction(legal, probabilities, random);
  return {
    action,
    probabilities,
    key: microInformationKey(game, game.currentPlayer),
    actionValues,
    rangeSize,
  };
}

export function microInformationKey(
  game: MicroGame,
  player: MicroPlayer,
): string {
  const board =
    game.street === 0 && !game.showdown
      ? "hidden"
      : game.board.map(microCardLabel).join("-");
  return (
    `${game.holeCards[player].map(microCardLabel).join("")}|${board}|` +
    `p${game.contributions[0] + game.contributions[1]}|` +
    `tc${microToCall(game, player)}|h${game.history.map((item) => item.action).join("")}`
  );
}

function resolveAction(
  game: MicroGame,
  model: BrowserStrategy,
  rolloutsPerAction: number,
  random: () => number,
): { action: MicroAction; values: number[]; rangeSize: number } {
  const observer = game.currentPlayer;
  const range = inferRange(game, observer, model, random);
  const sampledStates = Array.from({ length: rolloutsPerAction }, () => {
    const combo = weightedChoice(range.combos, range.weights, random);
    return counterfactualGame(game, observer, combo, random);
  });
  const values = Array.from({ length: ACTION_COUNT }, () => Number.NEGATIVE_INFINITY);
  for (const action of legalMicroActions(game)) {
    let total = 0;
    for (const sampled of sampledStates) {
      const child = applyMicroAction(sampled, action);
      total += rolloutValue(child, observer, model, random);
    }
    values[action] = total / sampledStates.length;
  }
  const action = legalMicroActions(game).reduce((best, candidate) =>
    values[candidate] > values[best] ? candidate : best,
  );
  return { action, values, rangeSize: range.combos.length };
}

function inferRange(
  game: MicroGame,
  observer: MicroPlayer,
  model: BrowserStrategy,
  random: () => number,
): { combos: Array<[number, number]>; weights: number[] } {
  const visibleBoard = game.street === 1 || game.showdown ? game.board : [];
  const known = new Set([...game.holeCards[observer], ...visibleBoard]);
  const available = Array.from({ length: 24 }, (_, card) => card).filter(
    (card) => !known.has(card),
  );
  const combos: Array<[number, number]> = [];
  const weights: number[] = [];
  for (let first = 0; first < available.length; first += 1) {
    for (let second = first + 1; second < available.length; second += 1) {
      const combo: [number, number] = [available[first], available[second]];
      let replay = counterfactualGame(
        { ...game, history: [], events: [] },
        observer,
        combo,
        random,
      );
      let likelihood = 1;
      for (const entry of game.history) {
        if (replay.currentPlayer !== observer) {
          likelihood *= Math.max(
            blueprintProbabilities(replay, model)[entry.action],
            1e-9,
          );
        }
        replay = applyMicroAction(replay, entry.action);
      }
      combos.push(combo);
      weights.push(likelihood);
    }
  }
  const total = weights.reduce((sum, value) => sum + value, 0);
  return {
    combos,
    weights: weights.map((value) => value / total),
  };
}

function counterfactualGame(
  game: MicroGame,
  observer: MicroPlayer,
  opponentCards: [number, number],
  random: () => number,
): MicroGame {
  const holes: [[number, number], [number, number]] = [
    [...game.holeCards[0]],
    [...game.holeCards[1]],
  ];
  holes[observer] = [...game.holeCards[observer]];
  holes[(1 - observer) as MicroPlayer] = opponentCards;
  const visible = game.street === 1 || game.showdown;
  let board: [number, number, number];
  if (visible) {
    board = [...game.board];
  } else {
    const excluded = new Set([...holes[0], ...holes[1]]);
    const available = Array.from({ length: 24 }, (_, card) => card).filter(
      (card) => !excluded.has(card),
    );
    shuffle(available, random);
    board = [available[0], available[1], available[2]];
  }
  let replay = initialMicroGame(game.handId, holes, board, game.button);
  for (const entry of game.history) replay = applyMicroAction(replay, entry.action);
  return replay;
}

function rolloutValue(
  game: MicroGame,
  observer: MicroPlayer,
  model: BrowserStrategy,
  random: () => number,
): number {
  let state = game;
  while (!isMicroTerminal(state)) {
    const probabilities = blueprintProbabilities(state, model);
    const action = sampleAction(legalMicroActions(state), probabilities, random);
    state = applyMicroAction(state, action);
  }
  return microTerminalPayoffs(state)[observer];
}

function initialMicroGame(
  handId: number,
  holeCards: [[number, number], [number, number]],
  board: [number, number, number],
  button: MicroPlayer,
): MicroGame {
  const contributions: [number, number] = [2, 2];
  contributions[button] = 1;
  return {
    handId,
    holeCards,
    board,
    button,
    currentPlayer: button,
    street: 0,
    contributions,
    streetContributions: [...contributions],
    acted: [false, false],
    raises: 0,
    foldedPlayer: null,
    showdown: false,
    history: [],
    events: [
      `Hand ${handId + 1} · You ${button === 0 ? "post 1 SB" : "post 2 BB"}`,
    ],
  };
}

function microChipsFor(game: MicroGame, action: MicroAction): number {
  const player = game.currentPlayer;
  const stack = microStacks(game)[player];
  const outstanding = Math.min(microToCall(game, player), stack);
  if (action === 1) return outstanding;
  if (action === 4) return stack;
  if (action !== 2 && action !== 3) return 0;
  const potAfterCall =
    game.contributions[0] + game.contributions[1] + outstanding;
  const increment = Math.max(
    1,
    Math.ceil((action === 2 ? 0.5 : 1) * potAfterCall),
  );
  return Math.min(stack, outstanding + increment);
}

function evaluateFive(cards: number[]): number[] {
  const ranks = cards.map((card) => Math.floor(card / 4));
  const suits = cards.map((card) => card % 4);
  const counts = new Map<number, number>();
  for (const rank of ranks) counts.set(rank, (counts.get(rank) ?? 0) + 1);
  const groups = [...counts].sort(
    ([rankA, countA], [rankB, countB]) => countB - countA || rankB - rankA,
  );
  const unique = [...counts.keys()].sort((a, b) => a - b);
  const straight = unique.length === 5 && unique[4] - unique[0] === 4;
  const flush = new Set(suits).size === 1;
  if (straight && flush) return [8, unique[4]];
  if (groups[0][1] === 4) return [7, groups[0][0], groups[1][0]];
  if (groups[0][1] === 3 && groups[1][1] === 2) {
    return [6, groups[0][0], groups[1][0]];
  }
  if (flush) return [5, ...[...ranks].sort((a, b) => b - a)];
  if (straight) return [4, unique[4]];
  if (groups[0][1] === 3) {
    return [3, groups[0][0], ...groups.slice(1).map(([rank]) => rank).sort((a, b) => b - a)];
  }
  const pairs = groups.filter(([, count]) => count === 2).map(([rank]) => rank);
  if (pairs.length === 2) {
    const kicker = groups.find(([, count]) => count === 1)?.[0] ?? -1;
    return [2, ...pairs.sort((a, b) => b - a), kicker];
  }
  if (pairs.length === 1) {
    return [
      1,
      pairs[0],
      ...groups
        .filter(([, count]) => count === 1)
        .map(([rank]) => rank)
        .sort((a, b) => b - a),
    ];
  }
  return [0, ...[...ranks].sort((a, b) => b - a)];
}

function canonicalCards(
  hole: [number, number],
  board: number[],
): [[number, number], number[]] {
  let best: number[] | null = null;
  for (const suitMap of SUIT_PERMUTATIONS) {
    const transform = (card: number) =>
      Math.floor(card / 4) * 4 + suitMap[card % 4];
    const transformed = [
      ...hole.map(transform).sort((a, b) => a - b),
      ...board.map(transform).sort((a, b) => a - b),
    ];
    if (!best || compareTuple(transformed, best) < 0) best = transformed;
  }
  if (!best) throw new Error("Suit canonicalization failed");
  return [[best[0], best[1]], best.slice(2)];
}

function compareTuple(first: number[], second: number[]): number {
  for (let index = 0; index < Math.max(first.length, second.length); index += 1) {
    const difference = (first[index] ?? -1) - (second[index] ?? -1);
    if (difference !== 0) return difference;
  }
  return 0;
}

function uniformProbabilities(legal: MicroAction[]): number[] {
  const probabilities = Array.from({ length: ACTION_COUNT }, () => 0);
  for (const action of legal) probabilities[action] = 1 / legal.length;
  return probabilities;
}

function sampleAction(
  legal: MicroAction[],
  probabilities: number[],
  random: () => number,
): MicroAction {
  const draw = random();
  let cumulative = 0;
  for (const action of legal) {
    cumulative += probabilities[action];
    if (draw <= cumulative) return action;
  }
  return legal[legal.length - 1];
}

function weightedChoice<T>(
  items: T[],
  weights: number[],
  random: () => number,
): T {
  const draw = random();
  let cumulative = 0;
  for (let index = 0; index < items.length; index += 1) {
    cumulative += weights[index];
    if (draw <= cumulative) return items[index];
  }
  return items[items.length - 1];
}

function shuffle(values: number[], random: () => number): void {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [values[index], values[swap]] = [values[swap], values[index]];
  }
}

function permutations(values: number[]): number[][] {
  if (values.length === 0) return [[]];
  return values.flatMap((value, index) =>
    permutations([...values.slice(0, index), ...values.slice(index + 1)]).map(
      (rest) => [value, ...rest],
    ),
  );
}

const SUIT_PERMUTATIONS = permutations([0, 1, 2, 3]);
