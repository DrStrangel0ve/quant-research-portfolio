export type PokerAction = 0 | 1 | 2 | 3;
export type Player = 0 | 1;

export const ACTION_NAMES: Record<PokerAction, string> = {
  0: "Call",
  1: "Raise",
  2: "Fold",
  3: "Check",
};

export interface LeducGame {
  handId: number;
  privateCards: [number, number];
  publicCard: number;
  smallBlind: Player;
  currentPlayer: Player;
  roundIndex: 0 | 1;
  contributions: [number, number];
  raises: number;
  nonRaiseActions: number;
  foldedPlayer: Player | null;
  showdown: boolean;
  history: PokerAction[];
  events: string[];
}

export function createGame(handId: number, random = Math.random): LeducGame {
  const deck = [0, 1, 2, 3, 4, 5];
  for (let index = deck.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [deck[index], deck[swap]] = [deck[swap], deck[index]];
  }
  const smallBlind = (handId % 2) as Player;
  const contributions: [number, number] =
    smallBlind === 0 ? [1, 2] : [2, 1];
  return {
    handId,
    privateCards: [deck[0], deck[1]],
    publicCard: deck[2],
    smallBlind,
    currentPlayer: smallBlind,
    roundIndex: 0,
    contributions,
    raises: 0,
    nonRaiseActions: 0,
    foldedPlayer: null,
    showdown: false,
    history: [],
    events: [
      `Hand ${handId + 1} · You ${smallBlind === 0 ? "post 1 SB" : "post 2 BB"}`,
    ],
  };
}

export function isTerminal(game: LeducGame): boolean {
  return game.foldedPlayer !== null || game.showdown;
}

export function legalActions(game: LeducGame): PokerAction[] {
  if (isTerminal(game)) return [];
  const player = game.currentPlayer;
  const opponent = (1 - player) as Player;
  const facingBet = game.contributions[player] < game.contributions[opponent];
  const actions: PokerAction[] = [];
  if (facingBet) actions.push(0);
  if (game.raises < 2) actions.push(1);
  actions.push(2);
  if (!facingBet) actions.push(3);
  return actions;
}

export function applyAction(game: LeducGame, action: PokerAction): LeducGame {
  if (!legalActions(game).includes(action)) {
    throw new Error(`${ACTION_NAMES[action]} is not legal in this state`);
  }
  const player = game.currentPlayer;
  const opponent = (1 - player) as Player;
  const contributions: [number, number] = [...game.contributions];
  let raises = game.raises;
  let nonRaiseActions = game.nonRaiseActions;
  const actor = player === 0 ? "You" : "Bot";
  const events = [
    ...game.events,
    `${game.roundIndex === 0 ? "Pre" : "Post"} · ${actor} ${ACTION_NAMES[action].toLowerCase()}`,
  ];
  const history = [...game.history, action];

  if (action === 0) {
    contributions[player] = contributions[opponent];
    nonRaiseActions += 1;
  } else if (action === 1) {
    contributions[player] =
      Math.max(...contributions) + (game.roundIndex === 0 ? 2 : 4);
    raises += 1;
    nonRaiseActions = 1;
  } else if (action === 2) {
    return {
      ...game,
      currentPlayer: opponent,
      foldedPlayer: player,
      history,
      events,
    };
  } else {
    nonRaiseActions += 1;
  }

  const advanced: LeducGame = {
    ...game,
    currentPlayer: opponent,
    contributions,
    raises,
    nonRaiseActions,
    history,
    events,
  };
  if (nonRaiseActions < 2) return advanced;
  if (game.roundIndex === 0) {
    return {
      ...advanced,
      roundIndex: 1,
      raises: 0,
      nonRaiseActions: 0,
      events: [...events, `Board · ${cardLabel(game.publicCard)} revealed`],
    };
  }
  return { ...advanced, showdown: true };
}

export function terminalPayoffs(game: LeducGame): [number, number] {
  if (!isTerminal(game)) throw new Error("The hand is not over");
  const pot = game.contributions[0] + game.contributions[1];
  let winners: Player[];
  if (game.foldedPlayer !== null) {
    winners = [(1 - game.foldedPlayer) as Player];
  } else {
    const scores = ([0, 1] as Player[]).map((player) => {
      const privateRank = cardRank(game.privateCards[player]);
      return [Number(privateRank === cardRank(game.publicCard)), privateRank];
    });
    const comparison = compareScore(scores[0], scores[1]);
    winners = comparison === 0 ? [0, 1] : [comparison > 0 ? 0 : 1];
  }
  const shares: [number, number] = [0, 0];
  for (const winner of winners) shares[winner] = pot / winners.length;
  return [
    (shares[0] - game.contributions[0]) / 2,
    (shares[1] - game.contributions[1]) / 2,
  ];
}

export function informationKey(
  game: LeducGame,
  player: Player,
  perfectRecall: boolean,
): string {
  const opponent = (1 - player) as Player;
  const board = game.roundIndex === 0 ? -1 : cardRank(game.publicCard);
  const compact =
    `h${cardRank(game.privateCards[player])}|b${board}|` +
    `m${game.contributions[player]}|o${game.contributions[opponent]}`;
  if (!perfectRecall) return compact;
  return `${compact}|r${game.roundIndex}|t${game.history.join(",")}`;
}

export function cardLabel(card: number): string {
  const ranks = ["J", "Q", "K"];
  return `${ranks[cardRank(card)]}${card % 2 === 0 ? "♠" : "♥"}`;
}

export function cardRank(card: number): number {
  return Math.floor(card / 2);
}

function compareScore(first: number[], second: number[]): number {
  if (first[0] !== second[0]) return first[0] - second[0];
  return first[1] - second[1];
}
