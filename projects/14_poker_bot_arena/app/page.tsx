"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ACTION_NAMES,
  applyAction,
  cardLabel,
  createGame,
  informationKey,
  isTerminal,
  legalActions,
  terminalPayoffs,
  type LeducGame,
  type PokerAction,
} from "./game";

type BotId =
  | "cfr"
  | "rlcard"
  | "punisher"
  | "calling"
  | "aggressive"
  | "random";

type PolicyTable = Record<string, number[]>;

interface ArenaPolicies {
  trained: {
    information_mode: "perfect_recall";
    policy: PolicyTable;
  };
  rlcard_reference: {
    information_mode: "compact";
    policy: PolicyTable;
  };
  exploit_bot: {
    actions_by_seat: Record<"0" | "1", Record<string, PokerAction>>;
  };
}

interface Decision {
  action: PokerAction;
  probabilities: number[];
  key: string;
}

const BOTS: Record<
  BotId,
  { name: string; tag: string; description: string; accent: string }
> = {
  cfr: {
    name: "CFR+ 20K",
    tag: "TRAINED",
    description: "288-state perfect-recall average policy. Exact exploitability: 0.042 BB/hand.",
    accent: "#f26b21",
  },
  rlcard: {
    name: "RLCard CFR",
    tag: "REFERENCE",
    description: "Published 84-state compact CFR checkpoint from RLCard 1.2.0.",
    accent: "#b7c8a7",
  },
  punisher: {
    name: "Exact Punisher",
    tag: "ADVERSARY",
    description: "Full-history best response tuned to attack the CFR+ 20K policy.",
    accent: "#f0c674",
  },
  aggressive: {
    name: "Pressure Bot",
    tag: "HEURISTIC",
    description: "Raises whenever the cap allows, then calls. Simple and volatile.",
    accent: "#df8164",
  },
  calling: {
    name: "Calling Station",
    tag: "HEURISTIC",
    description: "Never folds or raises. A clean test of value betting.",
    accent: "#81ad9b",
  },
  random: {
    name: "Chaos Bot",
    tag: "BASELINE",
    description: "Uniform random play across every legal action.",
    accent: "#9e9aa7",
  },
};

const ACTION_SHORTCUTS: Partial<Record<string, PokerAction>> = {
  c: 0,
  r: 1,
  f: 2,
  k: 3,
};

export default function Home() {
  const [policies, setPolicies] = useState<ArenaPolicies | null>(null);
  const [selectedBot, setSelectedBot] = useState<BotId>("cfr");
  const [game, setGame] = useState<LeducGame>(() => createGame(0));
  const [score, setScore] = useState(0);
  const [hands, setHands] = useState(0);
  const [readout, setReadout] = useState<Decision | null>(null);
  const settledHand = useRef<number | null>(null);

  useEffect(() => {
    fetch("/poker-policies.json")
      .then((response) => {
        if (!response.ok) throw new Error("Policy checkpoint failed to load");
        return response.json();
      })
      .then((payload: ArenaPolicies) => setPolicies(payload))
      .catch(() => setPolicies(null));
  }, []);

  useEffect(() => {
    if (!isTerminal(game) || settledHand.current === game.handId) return;
    settledHand.current = game.handId;
    setScore((current) => current + terminalPayoffs(game)[0]);
    setHands((current) => current + 1);
  }, [game]);

  const act = useCallback((action: PokerAction) => {
    setGame((current) => {
      if (isTerminal(current) || current.currentPlayer !== 0) return current;
      if (!legalActions(current).includes(action)) return current;
      return applyAction(current, action);
    });
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const action = ACTION_SHORTCUTS[event.key.toLowerCase()];
      if (action !== undefined) act(action);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [act]);

  useEffect(() => {
    if (isTerminal(game) || game.currentPlayer !== 1) return;
    const timer = window.setTimeout(() => {
      setGame((current) => {
        if (isTerminal(current) || current.currentPlayer !== 1) return current;
        const decision = chooseBotAction(current, selectedBot, policies);
        setReadout(decision);
        return applyAction(current, decision.action);
      });
    }, 520);
    return () => window.clearTimeout(timer);
  }, [game, policies, selectedBot]);

  const startNextHand = () => {
    const nextId = game.handId + 1;
    settledHand.current = null;
    setReadout(null);
    setGame(createGame(nextId));
  };

  const changeBot = (bot: BotId) => {
    setSelectedBot(bot);
    setScore(0);
    setHands(0);
    settledHand.current = null;
    setReadout(null);
    setGame(createGame(game.handId + 1));
  };

  const result = isTerminal(game) ? terminalPayoffs(game)[0] : null;
  const legal = legalActions(game);
  const bot = BOTS[selectedBot];
  const pot = game.contributions[0] + game.contributions[1];
  const thinking = !isTerminal(game) && game.currentPlayer === 1;
  const status = isTerminal(game)
    ? result && result > 0
      ? `You win +${result.toFixed(2)} BB`
      : result && result < 0
        ? `Bot wins ${Math.abs(result).toFixed(2)} BB`
        : "Split pot"
    : thinking
      ? `${bot.name} is solving…`
      : game.currentPlayer === 0
        ? "Your decision"
        : "Bot decision";

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#table" aria-label="Poker Lab home">
          <span className="brand-mark">PL</span>
          <span>POKER LAB</span>
        </a>
        <div className="header-proof">
          <span><i className="live-dot" /> CHECKPOINT LOADED</span>
          <span>20,000 ITERATIONS</span>
          <span>0.042 EXPLOITABILITY</span>
        </div>
        <a
          className="source-link"
          href="https://github.com/DrStrangel0ve/quant-research-portfolio"
          target="_blank"
          rel="noreferrer"
        >
          VIEW SOURCE <span aria-hidden="true">↗</span>
        </a>
      </header>

      <section className="hero-copy">
        <p className="eyebrow">IMPERFECT INFORMATION · EXACTLY AUDITED</p>
        <h1>Play the policy.<br /><em>Find the leak.</em></h1>
        <p className="lede">
          Heads-up Leduc Hold&apos;em against a from-scratch CFR+ agent,
          a published reference checkpoint, and four interpretable opponents.
        </p>
      </section>

      <section className="arena-shell" id="table">
        <aside className="bot-roster" aria-label="Choose an opponent">
          <div className="panel-heading">
            <span>OPPONENTS</span>
            <b>06</b>
          </div>
          {(Object.keys(BOTS) as BotId[]).map((id) => (
            <button
              key={id}
              className={`bot-option ${selectedBot === id ? "selected" : ""}`}
              onClick={() => changeBot(id)}
              style={{ "--bot-accent": BOTS[id].accent } as React.CSSProperties}
              aria-pressed={selectedBot === id}
            >
              <span className="bot-index">0{Object.keys(BOTS).indexOf(id) + 1}</span>
              <span>
                <strong>{BOTS[id].name}</strong>
                <small>{BOTS[id].tag}</small>
              </span>
              <i aria-hidden="true">→</i>
            </button>
          ))}
          <div className="bot-note">
            <span style={{ background: bot.accent }} />
            <p>{bot.description}</p>
          </div>
        </aside>

        <div className="table-column">
          <div className="match-strip">
            <span>MATCH / {bot.name.toUpperCase()}</span>
            <strong className={score >= 0 ? "positive" : "negative"}>
              {score >= 0 ? "+" : ""}{score.toFixed(2)} BB
            </strong>
            <span>{hands} HAND{hands === 1 ? "" : "S"}</span>
          </div>

          <div className="poker-table">
            <div className="table-grid" aria-hidden="true" />
            <PlayerSeat
              label={bot.name}
              contribution={game.contributions[1]}
              card={game.privateCards[1]}
              hidden={!isTerminal(game)}
              active={!isTerminal(game) && game.currentPlayer === 1}
              position={game.smallBlind === 1 ? "SB" : "BB"}
              className="bot-seat"
            />

            <div className="board-zone">
              <div className="pot-label">POT <strong>{pot}</strong></div>
              <PlayingCard
                card={game.publicCard}
                hidden={game.roundIndex === 0 && !game.showdown}
                board
              />
              <span className="round-label">
                {game.roundIndex === 0 ? "PRE-REVEAL" : "PUBLIC CARD"}
              </span>
            </div>

            <PlayerSeat
              label="YOU"
              contribution={game.contributions[0]}
              card={game.privateCards[0]}
              hidden={false}
              active={!isTerminal(game) && game.currentPlayer === 0}
              position={game.smallBlind === 0 ? "SB" : "BB"}
              className="human-seat"
            />

            <div className={`decision-status ${isTerminal(game) ? "terminal" : ""}`}>
              <i />
              {status}
            </div>
          </div>

          <div className="action-deck">
            {isTerminal(game) ? (
              <button className="next-hand" onClick={startNextHand}>
                DEAL NEXT HAND <span>→</span>
              </button>
            ) : (
              ([2, 0, 3, 1] as PokerAction[]).map((action) => {
                const enabled = game.currentPlayer === 0 && legal.includes(action);
                return (
                  <button
                    key={action}
                    disabled={!enabled}
                    className={action === 1 ? "primary-action" : ""}
                    onClick={() => act(action)}
                  >
                    <kbd>{action === 0 ? "C" : action === 1 ? "R" : action === 2 ? "F" : "K"}</kbd>
                    {ACTION_NAMES[action]}
                    {action === 1 && enabled && (
                      <small>+{game.roundIndex === 0 ? 2 : 4}</small>
                    )}
                  </button>
                );
              })
            )}
          </div>
        </div>

        <aside className="telemetry">
          <div className="panel-heading">
            <span>POLICY TRACE</span>
            <b>{policies ? "LIVE" : "LOAD"}</b>
          </div>
          <div className="trace-card">
            <p>LAST BOT MIX</p>
            <ProbabilityBars decision={readout} />
          </div>
          <div className="trace-card">
            <p>HAND LEDGER</p>
            <ol className="hand-log">
              {[...game.events].reverse().slice(0, 7).map((event, index) => (
                <li key={`${event}-${index}`}>
                  <span>{String(game.events.length - index).padStart(2, "0")}</span>
                  {event}
                </li>
              ))}
            </ol>
          </div>
          <div className="state-key">
            <p>VISIBLE STATE</p>
            <code>{informationKey(game, game.currentPlayer, true)}</code>
          </div>
        </aside>
      </section>

      <section className="proof-section">
        <div>
          <p className="eyebrow">MEASURED, NOT MARKETED</p>
          <h2>A bot you can<br />actually interrogate.</h2>
        </div>
        <dl className="proof-grid">
          <div><dt>0.0424</dt><dd>EXACT EXPLOITABILITY<br />BB / HAND</dd></div>
          <div><dt>+0.136</dt><dd>EXACT EDGE VS RLCARD<br />EITHER SEAT</dd></div>
          <div><dt>288</dt><dd>PERFECT-RECALL<br />INFORMATION STATES</dd></div>
          <div><dt>240</dt><dd>CHANCE OUTCOMES<br />FULLY ENUMERATED</dd></div>
        </dl>
        <p className="method-note">
          Leduc keeps the hidden cards, bluffing, public reveal, and sequential
          betting that make poker difficult—inside a game small enough to audit
          with an exact best response. The probability bars above are read
          directly from the trained JSON checkpoint.
        </p>
      </section>

      <footer>
        <span>POKER LAB / QUANT RESEARCH PORTFOLIO</span>
        <span>EDUCATIONAL RESEARCH · NOT A WAGERING PRODUCT</span>
      </footer>
    </main>
  );
}

function PlayerSeat({
  label,
  contribution,
  card,
  hidden,
  active,
  position,
  className,
}: {
  label: string;
  contribution: number;
  card: number;
  hidden: boolean;
  active: boolean;
  position: string;
  className: string;
}) {
  return (
    <div className={`player-seat ${className} ${active ? "active" : ""}`}>
      <PlayingCard card={card} hidden={hidden} />
      <div>
        <span className="position-pill">{position}</span>
        <strong>{label}</strong>
        <small>{contribution} CHIPS COMMITTED</small>
      </div>
    </div>
  );
}

function PlayingCard({
  card,
  hidden,
  board = false,
}: {
  card: number;
  hidden: boolean;
  board?: boolean;
}) {
  const label = cardLabel(card);
  const red = label.includes("♥");
  return (
    <div
      className={`playing-card ${hidden ? "card-back" : ""} ${board ? "board-card" : ""} ${red ? "red-card" : ""}`}
      aria-label={hidden ? "Hidden card" : label}
    >
      {hidden ? <span>PL</span> : <><b>{label[0]}</b><i>{label[1]}</i></>}
    </div>
  );
}

function ProbabilityBars({ decision }: { decision: Decision | null }) {
  const legalProbabilities = useMemo(() => {
    if (!decision) return [];
    return ([0, 1, 2, 3] as PokerAction[])
      .filter((action) => decision.probabilities[action] > 0)
      .map((action) => ({
        action,
        probability: decision.probabilities[action],
      }));
  }, [decision]);
  if (!decision) {
    return <div className="empty-trace">Waiting for the bot&apos;s first decision.</div>;
  }
  return (
    <div className="probability-bars">
      {legalProbabilities.map(({ action, probability }) => (
        <div key={action}>
          <span>{ACTION_NAMES[action]}</span>
          <i><b style={{ width: `${probability * 100}%` }} /></i>
          <strong>{(probability * 100).toFixed(1)}%</strong>
        </div>
      ))}
      <code title={decision.key}>{decision.key}</code>
    </div>
  );
}

function chooseBotAction(
  game: LeducGame,
  bot: BotId,
  policies: ArenaPolicies | null,
): Decision {
  const legal = legalActions(game);
  let probabilities = [0, 0, 0, 0];
  let key = informationKey(game, 1, bot !== "rlcard");

  if (bot === "calling") {
    probabilities[legal.includes(0) ? 0 : 3] = 1;
  } else if (bot === "aggressive") {
    const preferred = ([1, 0, 3, 2] as PokerAction[]).find((action) =>
      legal.includes(action),
    );
    probabilities[preferred ?? legal[0]] = 1;
  } else if (bot === "random" || !policies) {
    for (const action of legal) probabilities[action] = 1 / legal.length;
  } else if (bot === "punisher") {
    key = informationKey(game, 1, true);
    const chosen = policies.exploit_bot.actions_by_seat["1"][key];
    probabilities[legal.includes(chosen) ? chosen : legal[0]] = 1;
  } else {
    const source = bot === "cfr" ? policies.trained : policies.rlcard_reference;
    const perfectRecall = source.information_mode === "perfect_recall";
    key = informationKey(game, 1, perfectRecall);
    const raw = source.policy[key] ?? [0, 0, 0, 0];
    probabilities = raw.map((value, action) =>
      legal.includes(action as PokerAction) ? Math.max(value, 0) : 0,
    );
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    if (total <= 0) {
      probabilities = probabilities.map((_, action) =>
        legal.includes(action as PokerAction) ? 1 / legal.length : 0,
      );
    } else {
      probabilities = probabilities.map((value) => value / total);
    }
  }

  const draw = Math.random();
  let cumulative = 0;
  let action = legal[legal.length - 1];
  for (const candidate of legal) {
    cumulative += probabilities[candidate];
    if (draw <= cumulative) {
      action = candidate;
      break;
    }
  }
  return { action, probabilities, key };
}
