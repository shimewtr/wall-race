import type { Level } from '../engine/ai';
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
} from '../engine/game';
import type { AiRequest } from './ai.worker';

type Mode = 'human-ai' | 'ai-human' | 'human-human' | 'ai-ai';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const boardEl = $<HTMLDivElement>('board');
const statusEl = $<HTMLParagraphElement>('status');
const recordEl = $<HTMLOListElement>('record');
const modeEl = $<HTMLSelectElement>('mode');
const levelEl = $<HTMLSelectElement>('level');
const bars = [$<HTMLDivElement>('bar-0'), $<HTMLDivElement>('bar-1')];

const PLAYER_NAMES = ['青', '橙'];

let history: GameState[] = [initialState()];
let moves: Move[] = [];
let mode: Mode = 'human-ai';
let thinking = false;
let requestId = 0;
// タッチ操作では 1 回目のタップでプレビュー、同じ場所の 2 回目で確定
let pendingWall: { x: number; y: number; o: Orientation } | null = null;

const worker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
worker.onmessage = (e: MessageEvent<{ id: number; move: Move }>) => {
  if (e.data.id !== requestId) return; // 新規対局・やり直しで古くなった結果は捨てる
  thinking = false;
  play(e.data.move);
};

const current = () => history[history.length - 1];

const isHuman = (p: Player): boolean =>
  mode === 'human-human' || (mode === 'human-ai' && p === 0) || (mode === 'ai-human' && p === 1);

function play(m: Move) {
  const s = current();
  if (!isLegal(s, m)) return;
  history.push(applyMove(s, m));
  moves.push(m);
  pendingWall = null;
  render();
  maybeAskAi();
}

function maybeAskAi() {
  const s = current();
  if (isOver(s) || isHuman(s.turn)) return;
  thinking = true;
  render();
  const req: AiRequest = { id: ++requestId, state: s, level: levelEl.value as Level };
  // AI どうしの対局は見やすいように少し間を置く
  const delay = mode === 'ai-ai' ? 400 : 0;
  setTimeout(() => {
    if (req.id === requestId) worker.postMessage(req);
  }, delay);
}

function newGame() {
  requestId++;
  thinking = false;
  mode = modeEl.value as Mode;
  history = [initialState()];
  moves = [];
  pendingWall = null;
  render();
  maybeAskAi();
}

function undo() {
  requestId++;
  thinking = false;
  pendingWall = null;
  // 人間の手番に戻るまで巻き戻す
  do {
    if (history.length <= 1) break;
    history.pop();
    moves.pop();
  } while (!isHuman(current().turn) && mode !== 'ai-ai');
  render();
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
  if (o === 'h') place(el, 2 * x + 1, 2 * y + 2, 3, 1);
  else place(el, 2 * x + 2, 2 * y + 1, 1, 3);
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
      place(cell, 2 * x + 1, 2 * y + 1);
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
      boardEl.appendChild(slotEl(2 * x + 1, 2 * y + 2, 'h', Math.min(x, WALL_SLOTS - 1), y, 'h'));
    }
  }
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < WALL_SLOTS; x++) {
      boardEl.appendChild(slotEl(2 * x + 2, 2 * y + 1, 'v', x, Math.min(y, WALL_SLOTS - 1), 'v'));
    }
  }

  for (let y = 0; y < WALL_SLOTS; y++) {
    for (let x = 0; x < WALL_SLOTS; x++) {
      const i = y * WALL_SLOTS + x;
      const isLast = last?.type === 'wall' && last.x === x && last.y === y;
      if (s.hWalls[i]) boardEl.appendChild(wallEl(x, y, 'h', isLast && last.o === 'h' ? 'last' : ''));
      if (s.vWalls[i]) boardEl.appendChild(wallEl(x, y, 'v', isLast && last.o === 'v' ? 'last' : ''));
    }
  }

  if (pendingWall) showPreview(pendingWall.x, pendingWall.y, pendingWall.o);

  for (const p of [0, 1] as const) {
    const who = mode === 'human-human' ? PLAYER_NAMES[p] : isHuman(p) ? 'あなた' : 'AI';
    const pips = Array.from({ length: WALLS_PER_PLAYER }, (_, i) =>
      `<span class="pip${i < s.wallsLeft[p] ? '' : ' used'}"></span>`).join('');
    bars[p].innerHTML = `<span class="name"><span class="dot p${p}"></span>${who}</span><span class="pips" aria-label="残り壁 ${s.wallsLeft[p]} 枚">${pips}</span>`;
    bars[p].classList.toggle('turn', s.turn === p && !isOver(s));
  }

  statusEl.textContent = statusText(s);

  recordEl.replaceChildren(
    ...Array.from({ length: Math.ceil(moves.length / 2) }, (_, i) => {
      const li = document.createElement('li');
      li.textContent = moves.slice(i * 2, i * 2 + 2).map(moveToString).join('  ');
      return li;
    }),
  );
  recordEl.parentElement!.scrollTop = recordEl.parentElement!.scrollHeight;
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
  return mode === 'human-human' ? `${name(s.turn)}の番` : isHuman(s.turn) ? 'あなたの番' : 'AI の番';
}

$('new-game').addEventListener('click', newGame);
$('undo').addEventListener('click', undo);
modeEl.addEventListener('change', newGame);
boardEl.addEventListener('pointerleave', () => {
  if (!pendingWall) hidePreview();
});

newGame();
