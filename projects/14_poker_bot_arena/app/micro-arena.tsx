"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  MICRO_ACTION_NAMES,
  applyMicroAction,
  chooseMicroBotAction,
  createMicroGame,
  isMicroTerminal,
  legalMicroActions,
  microCardLabel,
  microInformationKey,
  microStacks,
  microTerminalPayoffs,
  type BrowserStrategy,
  type MicroAction,
  type MicroBotId,
  type MicroDecision,
  type MicroGame,
} from "./micro-game";

const MICRO_BOTS: Record<
  MicroBotId,
  { name: string; tag: string; description: string; accent: string }
> = {
  blueprint: {
    name: "Deep CFR",
    tag: "NEURAL BLUEPRINT",
    description:
      "Browser inference from the trained 126-input average-strategy MLP.",
    accent: "#f26b21",
  },
  resolver: {
    name: "Range Resolver",
    tag: "SEARCH",
    description:
      "Bayes-filters your range, then uses stratified, common-random-number blueprint rollouts.",
    accent: "#f0c674",
  },
  pressure: {
    name: "Pot Pressure",
    tag: "HEURISTIC",
    description: "Prefers pot-sized aggression whenever the abstraction permits it.",
    accent: "#df8164",
  },
  calling: {
    name: "Calling Station",
    tag: "HEURISTIC",
    description: "Always checks or calls. Useful for probing value-betting behavior.",
    accent: "#81ad9b",
  },
  random: {
    name: "Chaos",
    tag: "BASELINE",
    description: "Uniform random play across the currently legal action set.",
    accent: "#9e9aa7",
  },
};

const MICRO_SHORTCUTS: Partial<Record<string, MicroAction>> = {
  f: 0,
  c: 1,
  h: 2,
  p: 3,
  a: 4,
};

export function MicroArena({ onSwitch }: { onSwitch: () => void }) {
  const [model, setModel] = useState<BrowserStrategy | null>(null);
  const [selectedBot, setSelectedBot] = useState<MicroBotId>("blueprint");
  const [game, setGame] = useState<MicroGame>(() => createMicroGame(0));
  const [score, setScore] = useState(0);
  const [hands, setHands] = useState(0);
  const [readout, setReadout] = useState<MicroDecision | null>(null);
  const settledHand = useRef<number | null>(null);

  useEffect(() => {
    fetch("/micro-strategy.json")
      .then((response) => {
        if (!response.ok) throw new Error("Neural checkpoint failed to load");
        return response.json();
      })
      .then((payload: BrowserStrategy) => setModel(payload))
      .catch(() => setModel(null));
  }, []);

  useEffect(() => {
    if (!isMicroTerminal(game) || settledHand.current === game.handId) return;
    settledHand.current = game.handId;
    setScore((current) => current + microTerminalPayoffs(game)[0]);
    setHands((current) => current + 1);
  }, [game]);

  const act = useCallback((action: MicroAction) => {
    setGame((current) => {
      if (isMicroTerminal(current) || current.currentPlayer !== 0) return current;
      if (!legalMicroActions(current).includes(action)) return current;
      return applyMicroAction(current, action);
    });
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const action = MICRO_SHORTCUTS[event.key.toLowerCase()];
      if (action !== undefined) act(action);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [act]);

  useEffect(() => {
    if (isMicroTerminal(game) || game.currentPlayer !== 1) return;
    const timer = window.setTimeout(() => {
      setGame((current) => {
        if (isMicroTerminal(current) || current.currentPlayer !== 1) return current;
        const decision = chooseMicroBotAction(current, selectedBot, model);
        setReadout(decision);
        return applyMicroAction(current, decision.action);
      });
    }, selectedBot === "resolver" ? 180 : 480);
    return () => window.clearTimeout(timer);
  }, [game, model, selectedBot]);

  const startNextHand = () => {
    settledHand.current = null;
    setReadout(null);
    setGame(createMicroGame(game.handId + 1));
  };

  const changeBot = (bot: MicroBotId) => {
    setSelectedBot(bot);
    setScore(0);
    setHands(0);
    settledHand.current = null;
    setReadout(null);
    setGame(createMicroGame(game.handId + 1));
  };

  const result = isMicroTerminal(game) ? microTerminalPayoffs(game)[0] : null;
  const legal = legalMicroActions(game);
  const bot = MICRO_BOTS[selectedBot];
  const pot = game.contributions[0] + game.contributions[1];
  const stacks = microStacks(game);
  const thinking = !isMicroTerminal(game) && game.currentPlayer === 1;
  const status = isMicroTerminal(game)
    ? result && result > 0
      ? `You win +${result.toFixed(2)} BB`
      : result && result < 0
        ? `Bot wins ${Math.abs(result).toFixed(2)} BB`
        : "Split pot"
    : thinking
      ? `${bot.name} is solving…`
      : "Your decision";

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#micro-table" aria-label="Poker Lab home">
          <span className="brand-mark">PL</span>
          <span>POKER LAB</span>
        </a>
        <div className="mode-switch" aria-label="Select poker benchmark">
          <button onClick={onSwitch}>LEDUC / EXACT</button>
          <button className="active">MICRO NL / NEURAL</button>
        </div>
        <a
          className="source-link"
          href="https://github.com/DrStrangel0ve/quant-research-portfolio/tree/main/projects/15_neural_poker_solver"
          target="_blank"
          rel="noreferrer"
        >
          VIEW METHOD <span aria-hidden="true">↗</span>
        </a>
      </header>

      <section className="hero-copy micro-hero">
        <p className="eyebrow">NO-LIMIT ABSTRACTION · NEURAL CFR · RANGE SEARCH</p>
        <h1>Scale the tree.<br /><em>Keep the audit.</em></h1>
        <p className="lede">
          Two-card, two-street Royal Micro Hold&apos;em against a learned
          blueprint, belief-aware rollout search, and interpretable baselines.
        </p>
      </section>

      <section className="arena-shell" id="micro-table">
        <aside className="bot-roster" aria-label="Choose a micro Hold'em opponent">
          <div className="panel-heading">
            <span>OPPONENTS</span>
            <b>05</b>
          </div>
          {(Object.keys(MICRO_BOTS) as MicroBotId[]).map((id, index) => (
            <button
              key={id}
              className={`bot-option ${selectedBot === id ? "selected" : ""}`}
              onClick={() => changeBot(id)}
              style={{ "--bot-accent": MICRO_BOTS[id].accent } as React.CSSProperties}
              aria-pressed={selectedBot === id}
            >
              <span className="bot-index">0{index + 1}</span>
              <span>
                <strong>{MICRO_BOTS[id].name}</strong>
                <small>{MICRO_BOTS[id].tag}</small>
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

          <div className="poker-table micro-table">
            <div className="table-grid" aria-hidden="true" />
            <MicroSeat
              label={bot.name}
              cards={game.holeCards[1]}
              hidden={!isMicroTerminal(game)}
              active={!isMicroTerminal(game) && game.currentPlayer === 1}
              position={game.button === 1 ? "BTN / SB" : "BB"}
              stack={stacks[1]}
              className="bot-seat"
            />

            <div className="board-zone">
              <div className="pot-label">POT <strong>{pot}</strong></div>
              <div className="micro-board">
                {game.board.map((card) => (
                  <MicroCard
                    card={card}
                    hidden={game.street === 0 && !game.showdown}
                    board
                    key={card}
                  />
                ))}
              </div>
              <span className="round-label">
                {game.street === 0 ? "PREFLOP / BOARD HIDDEN" : "FLOP"}
              </span>
            </div>

            <MicroSeat
              label="YOU"
              cards={game.holeCards[0]}
              hidden={false}
              active={!isMicroTerminal(game) && game.currentPlayer === 0}
              position={game.button === 0 ? "BTN / SB" : "BB"}
              stack={stacks[0]}
              className="human-seat"
            />
            <div className={`decision-status ${isMicroTerminal(game) ? "terminal" : ""}`}>
              <i />
              {status}
            </div>
          </div>

          <div className="action-deck micro-actions">
            {isMicroTerminal(game) ? (
              <button className="next-hand" onClick={startNextHand}>
                DEAL NEXT HAND <span>→</span>
              </button>
            ) : (
              ([0, 1, 2, 3, 4] as MicroAction[]).map((action) => {
                const enabled = game.currentPlayer === 0 && legal.includes(action);
                return (
                  <button
                    key={action}
                    disabled={!enabled}
                    className={action >= 2 ? "primary-action" : ""}
                    onClick={() => act(action)}
                  >
                    <kbd>{["F", "C", "H", "P", "A"][action]}</kbd>
                    {MICRO_ACTION_NAMES[action]}
                  </button>
                );
              })
            )}
          </div>
        </div>

        <aside className="telemetry">
          <div className="panel-heading">
            <span>NEURAL TRACE</span>
            <b>{model ? "126 → 128 → 128 → 5" : "LOAD"}</b>
          </div>
          <div className="trace-card">
            <p>LAST BOT MIX</p>
            <MicroProbabilityBars decision={readout} />
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
            <code>{microInformationKey(game, game.currentPlayer)}</code>
          </div>
        </aside>
      </section>

      <section className="proof-section">
        <div>
          <p className="eyebrow">A SCALING EXPERIMENT</p>
          <h2>Learn globally.<br />Search locally.</h2>
        </div>
        <dl className="proof-grid">
          <div><dt>24</dt><dd>CARDS<br />9 THROUGH ACE</dd></div>
          <div><dt>126</dt><dd>OBSERVABLE<br />INPUT FEATURES</dd></div>
          <div><dt>5</dt><dd>DISCRETE<br />NO-LIMIT ACTIONS</dd></div>
          <div><dt>{model?.training.completed_iterations ?? "—"}</dt><dd>EXTERNAL-SAMPLING<br />ITERATIONS</dd></div>
        </dl>
        <p className="method-note">
          The neural policy sees canonicalized private and public cards, stacks,
          pot, position, price, legal actions, and public history—never the
          opponent&apos;s cards. Range Resolver updates possible holdings from
          your observed actions before running local blueprint rollouts. This is
          SOTA-inspired research infrastructure, not a frontier HUNL claim.
        </p>
      </section>

      <footer>
        <span>POKER LAB / ROYAL MICRO HOLD&apos;EM</span>
        <span>EDUCATIONAL RESEARCH · NOT A WAGERING PRODUCT</span>
      </footer>
    </main>
  );
}

function MicroSeat({
  label,
  cards,
  hidden,
  active,
  position,
  stack,
  className,
}: {
  label: string;
  cards: [number, number];
  hidden: boolean;
  active: boolean;
  position: string;
  stack: number;
  className: string;
}) {
  return (
    <div className={`player-seat micro-seat ${className} ${active ? "active" : ""}`}>
      <div className="hole-cards">
        {cards.map((card) => <MicroCard card={card} hidden={hidden} key={card} />)}
      </div>
      <div>
        <span className="position-pill">{position}</span>
        <strong>{label}</strong>
        <small>{stack} CHIPS BEHIND</small>
      </div>
    </div>
  );
}

function MicroCard({
  card,
  hidden,
  board = false,
}: {
  card: number;
  hidden: boolean;
  board?: boolean;
}) {
  const label = microCardLabel(card);
  const red = label.includes("♥") || label.includes("♦");
  return (
    <div
      className={`playing-card ${hidden ? "card-back" : ""} ${board ? "board-card" : ""} ${red ? "red-card" : ""}`}
      aria-label={hidden ? "Hidden card" : label}
    >
      {hidden ? <span>PL</span> : <><b>{label[0]}</b><i>{label[1]}</i></>}
    </div>
  );
}

function MicroProbabilityBars({ decision }: { decision: MicroDecision | null }) {
  const rows = useMemo(() => {
    if (!decision) return [];
    return ([0, 1, 2, 3, 4] as MicroAction[])
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
    <div className="probability-bars micro-probabilities">
      {rows.map(({ action, probability }) => (
        <div key={action}>
          <span>{MICRO_ACTION_NAMES[action]}</span>
          <i><b style={{ width: `${probability * 100}%` }} /></i>
          <strong>{(probability * 100).toFixed(1)}%</strong>
        </div>
      ))}
      {decision.rangeSize && <small>{decision.rangeSize} RANGE COMBOS SCORED</small>}
      <code title={decision.key}>{decision.key}</code>
    </div>
  );
}
