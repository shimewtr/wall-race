import { type Level, WIN, maxDetour, recentCells } from '../engine/ai';
import {
  type GameState,
  type Move,
  type Orientation,
  type Player,
  SIZE,
  WALLS_PER_PLAYER,
  WALL_SLOTS,
  applyMove,
  initialState,
  isLegal,
  isOver,
  moveToString,
  pawnMoves,
  shortestDistance,
} from '../engine/game';
import { type AiRequest, type AiResponse, EVAL_DEPTH } from './ai.worker';
import { calibration, champion } from './champion';
import { winRate } from '../train/calibrate';

type Mode = 'human-ai' | 'ai-human' | 'human-human' | 'ai-ai';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const boardEl = $<HTMLDivElement>('board');
const statusEl = $<HTMLParagraphElement>('status');
const modeEl = $<HTMLSelectElement>('mode');
const levelEl = $<HTMLSelectElement>('level');
const bars = [$<HTMLDivElement>('bar-0'), $<HTMLDivElement>('bar-1')];

const PLAYER_NAMES = ['青', '橙'];

let history: GameState[] = [initialState()];
let moves: Move[] = [];
let mode: Mode = 'human-ai';
let thinking = false;
let requestId = 0;
// AI どうしの対局の再生状態。paused の間は勝手に進めない。redo は「1 手戻す」で戻した手（進めるときに打ち直す）
let paused = true;
let redo: Move[] = [];
// 「1 手進める」で AI に 1 手だけ考えさせているとき
let stepping = false;
// タッチ操作では 1 回目のタップでプレビュー、同じ場所の 2 回目で確定
let pendingWall: { x: number; y: number; o: Orientation } | null = null;

const worker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
worker.onmessage = (e: MessageEvent<AiResponse>) => {
  if (e.data.kind !== 'move' || e.data.id !== requestId) return; // 新規対局・やり直しで古くなった結果は捨てる
  thinking = false;
  stepping = false;
  play(e.data.move);
};

// 形勢判断は AI の手番の計算を邪魔しないよう別のワーカーで行う
const evalWorker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
let evalId = 0;
evalWorker.onmessage = (e: MessageEvent<AiResponse>) => {
  if (e.data.kind !== 'eval' || e.data.id !== evalId) return;
  renderEval(e.data.score);
};

const current = () => history[history.length - 1];

const isHuman = (p: Player): boolean =>
  mode === 'human-human' || (mode === 'human-ai' && p === 0) || (mode === 'ai-human' && p === 1);

// fromRedo: 「1 手進める」で戻した手を打ち直すとき（そのときは戻した手の続きを残す）
function play(m: Move, fromRedo = false) {
  const s = current();
  if (!isLegal(s, m)) return;
  history.push(applyMove(s, m));
  moves.push(m);
  if (!fromRedo) redo = [];
  pendingWall = null;
  render();
  maybeAskAi();
}

const isAiAi = () => mode === 'ai-ai';

// 1 人で後手を持つときだけ表示を 180 度回し、人間を常に手前にする。
// ゲーム内部の座標と棋譜は変えない。
const isFlipped = () => mode === 'ai-human';
const displayCell = (x: number, y: number) =>
  isFlipped() ? { x: SIZE - 1 - x, y: SIZE - 1 - y } : { x, y };
const displayWall = (x: number, y: number) =>
  isFlipped() ? { x: WALL_SLOTS - 1 - x, y: WALL_SLOTS - 1 - y } : { x, y };

function maybeAskAi() {
  const s = current();
  if (isOver(s) || isHuman(s.turn)) return;
  // AI どうしは一時停止中なら進めない（「1 手進める」のときだけ 1 手考える）
  if (isAiAi() && paused && !stepping) {
    render();
    return;
  }
  // 戻した手があれば、AI に考えさせずにそれを打ち直す
  if (isAiAi() && redo.length > 0) {
    const id = ++requestId;
    setTimeout(() => {
      if (id === requestId && redo.length > 0) {
        stepping = false;
        play(redo.pop()!, true);
      }
    }, stepping ? 0 : 400);
    return;
  }
  thinking = true;
  render();
  const req: AiRequest = {
    kind: 'move',
    id: ++requestId,
    state: s,
    level: levelEl.value as Level,
    recent: recentCells(history, s.turn),
  };
  // AI どうしの対局は見やすいように少し間を置く（1 手ずつ進めるときは待たない）
  const delay = isAiAi() && !stepping ? 400 : 0;
  setTimeout(() => {
    if (req.id === requestId) worker.postMessage(req);
  }, delay);
}

// start: AI どうしのとき、すぐに再生を始めるか（対戦形式を切り替えただけのときは始めない）
function newGame(start = true) {
  requestId++;
  thinking = false;
  stepping = false;
  mode = modeEl.value as Mode;
  history = [initialState()];
  moves = [];
  redo = [];
  paused = !start;
  pendingWall = null;
  render();
  maybeAskAi();
}

function undo() {
  requestId++;
  thinking = false;
  stepping = false;
  pendingWall = null;
  if (isAiAi()) {
    // AI どうしは 1 手だけ戻して一時停止する。戻した手は「1 手進める」で打ち直せる
    paused = true;
    if (history.length > 1) {
      history.pop();
      redo.push(moves.pop()!);
    }
    render();
    return;
  }
  // 人間の手番に戻るまで巻き戻す
  do {
    if (history.length <= 1) break;
    history.pop();
    moves.pop();
  } while (!isHuman(current().turn));
  render();
  maybeAskAi();
}

function togglePlay() {
  if (isOver(current())) return;
  paused = !paused;
  if (paused) {
    // 考え中の手は捨てて止める
    requestId++;
    thinking = false;
    render();
  } else {
    maybeAskAi();
  }
}

function stepForward() {
  if (isOver(current()) || thinking || stepping) return;
  paused = true;
  stepping = true;
  maybeAskAi();
}

// ---- 描画 ----

const place = (el: HTMLElement, col: number, row: number, colSpan = 1, rowSpan = 1) => {
  el.style.gridColumn = `${col} / span ${colSpan}`;
  el.style.gridRow = `${row} / span ${rowSpan}`;
};

function wallEl(x: number, y: number, o: Orientation, cls: string): HTMLDivElement {
  const el = document.createElement('div');
  el.className = `wall ${o} ${cls}`;
  const shown = displayWall(x, y);
  if (o === 'h') place(el, 2 * shown.x + 1, 2 * shown.y + 2, 3, 1);
  else place(el, 2 * shown.x + 2, 2 * shown.y + 1, 1, 3);
  return el;
}

let previewEl: HTMLDivElement | null = null;

function showPreview(x: number, y: number, o: Orientation) {
  hidePreview();
  const s = current();
  const ok = canHumanAct() && isLegal(s, { type: 'wall', x, y, o });
  previewEl = wallEl(x, y, o, ok ? 'preview' : 'preview invalid');
  boardEl.appendChild(previewEl);
}

function hidePreview() {
  previewEl?.remove();
  previewEl = null;
}

const canHumanAct = () => !thinking && !isOver(current()) && isHuman(current().turn);

function slotEl(col: number, row: number, cls: string, x: number, y: number, o: Orientation) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = `slot ${cls}`;
  el.setAttribute('aria-label', `壁 ${moveToString({ type: 'wall', x, y, o })}`);
  place(el, col, row);
  el.addEventListener('pointerenter', (e) => {
    if (e.pointerType === 'mouse') showPreview(x, y, o);
  });
  el.addEventListener('pointerleave', (e) => {
    if (e.pointerType === 'mouse') hidePreview();
  });
  el.addEventListener('click', (e) => {
    if (!canHumanAct()) return;
    const isMouse = (e as PointerEvent).pointerType === 'mouse';
    const same = pendingWall && pendingWall.x === x && pendingWall.y === y && pendingWall.o === o;
    if (isMouse || same) {
      play({ type: 'wall', x, y, o });
    } else {
      pendingWall = { x, y, o };
      showPreview(x, y, o);
    }
  });
  return el;
}

function render() {
  const s = current();
  const last = moves[moves.length - 1];
  const targets = canHumanAct() ? pawnMoves(s) : [];
  boardEl.replaceChildren();
  previewEl = null;

  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'cell';
      if (y === 0) cell.classList.add('goal-0');
      if (y === SIZE - 1) cell.classList.add('goal-1');
      if (last?.type === 'pawn' && last.x === x && last.y === y) cell.classList.add('last');
      cell.setAttribute('aria-label', moveToString({ type: 'pawn', x, y }));
      const shown = displayCell(x, y);
      place(cell, 2 * shown.x + 1, 2 * shown.y + 1);
      if (targets.some((t) => t.x === x && t.y === y)) {
        cell.classList.add('target');
        cell.addEventListener('click', () => play({ type: 'pawn', x, y }));
      }
      s.pawns.forEach((p, i) => {
        if (p.x === x && p.y === y) {
          const pawn = document.createElement('span');
          pawn.className = `pawn p${i}${s.turn === i && !isOver(s) ? ' active' : ''}`;
          cell.appendChild(pawn);
        }
      });
      boardEl.appendChild(cell);
    }
  }

  // 壁を置くための溝。横の溝は横壁、縦の溝は縦壁。端の溝は内側にずらしたアンカーに対応させる。
  for (let y = 0; y < WALL_SLOTS; y++) {
    for (let x = 0; x < SIZE; x++) {
      const shown = displayCell(x, y);
      const displayGapY = isFlipped() ? shown.y : shown.y + 1;
      boardEl.appendChild(slotEl(2 * shown.x + 1, 2 * displayGapY, 'h', Math.min(x, WALL_SLOTS - 1), y, 'h'));
    }
  }
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < WALL_SLOTS; x++) {
      const shown = displayCell(x, y);
      const displayGapX = isFlipped() ? shown.x : shown.x + 1;
      boardEl.appendChild(slotEl(2 * displayGapX, 2 * shown.y + 1, 'v', x, Math.min(y, WALL_SLOTS - 1), 'v'));
    }
  }

  // 壁はどちらが置いたかで色分けする（盤面には持ち主を持たないので棋譜からたどる）
  moves.forEach((m, i) => {
    if (m.type !== 'wall') return;
    const owner = history[i].turn;
    const isLast = i === moves.length - 1;
    boardEl.appendChild(wallEl(m.x, m.y, m.o, `owner-${owner}${isLast ? ' last' : ''}`));
  });

  if (pendingWall) showPreview(pendingWall.x, pendingWall.y, pendingWall.o);

  // 表示上の手前・奥だけを入れ替える。AI 同士などは従来どおり青が手前。
  const near: Player = isFlipped() ? 1 : 0;
  const far: Player = near === 0 ? 1 : 0;
  boardEl.before(bars[far]);
  boardEl.after(bars[near]);

  for (const p of [0, 1] as const) {
    const who = mode === 'human-human' ? PLAYER_NAMES[p] : isHuman(p) ? 'あなた' : 'AI';
    const pips = Array.from({ length: WALLS_PER_PLAYER }, (_, i) =>
      `<span class="pip${i < s.wallsLeft[p] ? '' : ' used'}"></span>`).join('');
    bars[p].innerHTML = `<span class="name"><span class="dot p${p}"></span>${who}</span><span class="pips" aria-label="残り壁 ${s.wallsLeft[p]} 枚">${pips}</span>`;
    bars[p].classList.toggle('turn', s.turn === p && !isOver(s));
  }

  statusEl.textContent = statusText(s);
  requestEval(s);
  renderControls(s);

}

// ---- 形勢の表示 ----

const evalEl = $<HTMLElement>('eval');
const showEvalEl = $<HTMLInputElement>('show-eval');
const EVAL_KEY = 'wall-race:show-eval';
try {
  showEvalEl.checked = localStorage.getItem(EVAL_KEY) !== 'off';
} catch {
  // 保存できない環境では毎回表示する
}

const playerLabel = (p: Player) =>
  mode === 'human-human' ? PLAYER_NAMES[p] : `${PLAYER_NAMES[p]}（${isHuman(p) ? 'あなた' : 'AI'}）`;

let evaluatedState: GameState | null = null;

function requestEval(s: GameState) {
  evalEl.classList.toggle('hidden', !showEvalEl.checked);
  if (!showEvalEl.checked || s === evaluatedState) return;
  evaluatedState = s;
  renderBreakdown(s);
  if (isOver(s)) {
    evalId++;
    renderEval(null);
    return;
  }
  $('eval-text').textContent = '判断中…';
  evalWorker.postMessage({ kind: 'eval', id: ++evalId, state: s } satisfies AiRequest);
}

// score は手番側から見た評価値（1 歩 = 10 点）。null は決着済み
function renderEval(score: number | null) {
  const s = current();
  let blue: number; // 青から見た評価値
  let text: string;
  if (score === null) {
    blue = s.winner === 0 ? WIN : s.winner === 1 ? -WIN : 0;
    text = s.winner === null ? '引き分け' : `${playerLabel(s.winner)}の勝ち`;
  } else {
    blue = s.turn === 0 ? score : -score;
    const leader = (blue > 0 ? 0 : 1) as Player;
    if (Math.abs(blue) > WIN / 2) text = `${playerLabel(leader)}の勝ちを読み切り`;
    else if (calibration) text = `${playerLabel(0)}の勝率 ${Math.round(blueWinRate(score, s.turn) * 100)}%`;
    else text = `評価値 ${blue > 0 ? '+' : ''}${Math.round(blue)}（${PLAYER_NAMES[0]}から見て）`;
  }
  let pct: number;
  if (Math.abs(blue) > WIN / 2) pct = blue > 0 ? 100 : 0;
  else if (calibration && score !== null) pct = blueWinRate(score, s.turn) * 100;
  else pct = 50 + 50 * Math.tanh(blue / 100);
  $('eval-fill').style.width = `${pct}%`;
  $('eval-text').textContent = text;
}

// 手番側から見た評価値 → 青の勝率
function blueWinRate(score: number, turn: Player): number {
  const mover = winRate(score, calibration!);
  return turn === 0 ? mover : 1 - mover;
}

function renderBreakdown(s: GameState) {
  const cells = (f: (p: Player) => string) => ([0, 1] as const).map((p) => `<td>${f(p)}</td>`).join('');
  $('eval-table').innerHTML = `
    <tr><th></th>${([0, 1] as const).map((p) => `<th><span class="dot p${p}"></span>${PLAYER_NAMES[p]}</th>`).join('')}</tr>
    <tr><th>ゴールまでの残り歩数</th>${cells((p) => String(shortestDistance(s, p)))}</tr>
    <tr><th>残りの壁</th>${cells((p) => `${s.wallsLeft[p]} 枚`)}</tr>
    <tr><th>壁 1 枚で相手を遠回りさせられる歩数</th>${cells((p) => (s.wallsLeft[p] > 0 ? `${maxDetour(s, p)} 歩` : '—'))}</tr>`;
}

showEvalEl.addEventListener('change', () => {
  try {
    localStorage.setItem(EVAL_KEY, showEvalEl.checked ? 'on' : 'off');
  } catch {
    // 保存できなくても表示の切り替えはできる
  }
  evaluatedState = null;
  requestEval(current());
});

function renderControls(s: GameState) {
  const aiai = isAiAi();
  $('playback').hidden = !aiai;
  $('undo').hidden = aiai;
  const over = isOver(s);
  $<HTMLButtonElement>('play-pause').textContent = paused || over ? '▶ 再生' : '⏸ 一時停止';
  $<HTMLButtonElement>('play-pause').disabled = !aiai || over || stepping;
  $<HTMLButtonElement>('step-forward').disabled = !aiai || over || thinking || stepping;
  $<HTMLButtonElement>('step-back').disabled = !aiai || thinking || stepping || history.length <= 1;
}

function statusText(s: GameState): string {
  const name = (p: Player) =>
    mode === 'human-human' ? PLAYER_NAMES[p] : isHuman(p) ? 'あなた' : 'AI';
  if (s.winner !== null) {
    if (mode === 'human-ai' || mode === 'ai-human') return isHuman(s.winner) ? 'あなたの勝ち！' : 'AI の勝ち';
    return `${name(s.winner)}の勝ち`;
  }
  if (s.draw) return '引き分け（手数上限）';
  if (thinking) return 'AI が考え中…';
  if (isAiAi() && paused) return moves.length === 0 ? '' : '一時停止中';
  return mode === 'human-human' ? `${name(s.turn)}の番` : isHuman(s.turn) ? 'あなたの番' : 'AI の番';
}

{
  $('eval-note').textContent = calibration
    ? `第 ${calibration.generation} 世代の AI が ${EVAL_DEPTH} 手読みで判断した形勢を、同じ AI の自己対戦の勝ち負けから勝率に換算したもの`
    : `${champion ? `第 ${champion.generation} 世代` : '初期設定'}の AI が ${EVAL_DEPTH} 手読みで判断した評価値（勝率への換算式はまだ無い）`;
}

$('new-game').addEventListener('click', () => newGame(true));
$('undo').addEventListener('click', undo);
$('step-back').addEventListener('click', undo);
$('play-pause').addEventListener('click', togglePlay);
$('step-forward').addEventListener('click', stepForward);
// 対戦形式を切り替えただけでは AI どうしの対局を始めない
modeEl.addEventListener('change', () => newGame(false));
boardEl.addEventListener('pointerleave', () => {
  if (!pendingWall) hidePreview();
});

newGame(false);
