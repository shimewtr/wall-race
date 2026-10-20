// 壁で道をふさぎ合うレースゲームのルールエンジン。
// DOM に依存しないので、ブラウザでも Node（自己対戦・学習）でも同じコードが動く。

export const SIZE = 9;
export const WALL_SLOTS = SIZE - 1; // 壁のアンカーは 0..7 × 0..7
export const WALLS_PER_PLAYER = 10;
export const MAX_PLIES = 300; // 自己対戦で無限ループしないための上限（到達したら引き分け）

export type Player = 0 | 1;
export type Orientation = 'h' | 'v';

export interface Pos {
  x: number;
  y: number;
}

export type Move =
  | { type: 'pawn'; x: number; y: number }
  | { type: 'wall'; x: number; y: number; o: Orientation };

export interface GameState {
  pawns: [Pos, Pos];
  wallsLeft: [number, number];
  // アンカー (x, y) の壁。h はマス (x,y)(x+1,y) の下辺、v はマス (x,y)(x,y+1) の右辺をふさぐ。
  hWalls: Uint8Array;
  vWalls: Uint8Array;
  turn: Player;
  ply: number;
  winner: Player | null;
  draw: boolean;
}

// プレイヤー 0 は下から上（y=0 がゴール）、プレイヤー 1 は上から下（y=8 がゴール）
export const goalRow = (p: Player): number => (p === 0 ? 0 : SIZE - 1);

export function initialState(): GameState {
  return {
    pawns: [
      { x: 4, y: SIZE - 1 },
      { x: 4, y: 0 },
    ],
    wallsLeft: [WALLS_PER_PLAYER, WALLS_PER_PLAYER],
    hWalls: new Uint8Array(WALL_SLOTS * WALL_SLOTS),
    vWalls: new Uint8Array(WALL_SLOTS * WALL_SLOTS),
    turn: 0,
    ply: 0,
    winner: null,
    draw: false,
  };
}

export function cloneState(s: GameState): GameState {
  return {
    pawns: [{ ...s.pawns[0] }, { ...s.pawns[1] }],
    wallsLeft: [s.wallsLeft[0], s.wallsLeft[1]],
    hWalls: s.hWalls.slice(),
    vWalls: s.vWalls.slice(),
    turn: s.turn,
    ply: s.ply,
    winner: s.winner,
    draw: s.draw,
  };
}

export const isOver = (s: GameState): boolean => s.winner !== null || s.draw;

const inBoard = (x: number, y: number): boolean => x >= 0 && x < SIZE && y >= 0 && y < SIZE;
const inSlots = (x: number, y: number): boolean => x >= 0 && x < WALL_SLOTS && y >= 0 && y < WALL_SLOTS;
const wallAt = (walls: Uint8Array, x: number, y: number): boolean =>
  inSlots(x, y) && walls[y * WALL_SLOTS + x] === 1;

// 上下左右
export const DIRS: ReadonlyArray<readonly [number, number]> = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

// (x,y) から (x+dx, y+dy) へ 1 マス動けるか（盤外・壁を考慮。駒は考慮しない）
export function canStep(s: GameState, x: number, y: number, dx: number, dy: number): boolean {
  const nx = x + dx;
  const ny = y + dy;
  if (!inBoard(nx, ny)) return false;
  if (dy === -1) return !wallAt(s.hWalls, x, ny) && !wallAt(s.hWalls, x - 1, ny);
  if (dy === 1) return !wallAt(s.hWalls, x, y) && !wallAt(s.hWalls, x - 1, y);
  if (dx === -1) return !wallAt(s.vWalls, nx, y) && !wallAt(s.vWalls, nx, y - 1);
  return !wallAt(s.vWalls, x, y) && !wallAt(s.vWalls, x, y - 1);
}

// 駒の移動先（相手を飛び越える・斜めに回り込むルールを含む）
export function pawnMoves(s: GameState, p: Player = s.turn): Pos[] {
  const me = s.pawns[p];
  const op = s.pawns[1 - p];
  const result: Pos[] = [];
  for (const [dx, dy] of DIRS) {
    if (!canStep(s, me.x, me.y, dx, dy)) continue;
    const nx = me.x + dx;
    const ny = me.y + dy;
    if (nx !== op.x || ny !== op.y) {
      result.push({ x: nx, y: ny });
      continue;
    }
    // 相手が隣にいる: まっすぐ飛び越えられればそれ、無理なら斜め
    if (canStep(s, op.x, op.y, dx, dy)) {
      result.push({ x: op.x + dx, y: op.y + dy });
      continue;
    }
    const sides: Array<[number, number]> = dx === 0 ? [[-1, 0], [1, 0]] : [[0, -1], [0, 1]];
    for (const [sx, sy] of sides) {
      if (canStep(s, op.x, op.y, sx, sy)) result.push({ x: op.x + sx, y: op.y + sy });
    }
  }
  return result;
}

// ゴール行までの最短歩数（駒は無視、壁のみ考慮）。到達不能なら -1。
// 探索の中で何万回も呼ばれるので、経路は作らず距離だけを求め、作業用の配列も使い回す
const DIST = new Int16Array(SIZE * SIZE);
const QUEUE = new Int16Array(SIZE * SIZE);

export function shortestDistance(s: GameState, p: Player): number {
  const start = s.pawns[p];
  const goal = goalRow(p);
  if (start.y === goal) return 0;
  DIST.fill(-1);
  const startIdx = start.y * SIZE + start.x;
  DIST[startIdx] = 0;
  let head = 0;
  let tail = 0;
  QUEUE[tail++] = startIdx;
  while (head < tail) {
    const idx = QUEUE[head++];
    const x = idx % SIZE;
    const y = (idx - x) / SIZE;
    const nextDistance = DIST[idx] + 1;
    let nIdx: number;

    // 上。canStep と同じ壁判定を9×9固定盤向けに直接展開する。
    if (
      y > 0 &&
      (x === WALL_SLOTS || !s.hWalls[(y - 1) * WALL_SLOTS + x]) &&
      (x === 0 || !s.hWalls[(y - 1) * WALL_SLOTS + x - 1])
    ) {
      nIdx = idx - SIZE;
      if (DIST[nIdx] === -1) {
        DIST[nIdx] = nextDistance;
        if (y - 1 === goal) return nextDistance;
        QUEUE[tail++] = nIdx;
      }
    }

    // 下
    if (
      y < WALL_SLOTS &&
      (x === WALL_SLOTS || !s.hWalls[y * WALL_SLOTS + x]) &&
      (x === 0 || !s.hWalls[y * WALL_SLOTS + x - 1])
    ) {
      nIdx = idx + SIZE;
      if (DIST[nIdx] === -1) {
        DIST[nIdx] = nextDistance;
        if (y + 1 === goal) return nextDistance;
        QUEUE[tail++] = nIdx;
      }
    }

    // 左
    if (
      x > 0 &&
      (y === WALL_SLOTS || !s.vWalls[y * WALL_SLOTS + x - 1]) &&
      (y === 0 || !s.vWalls[(y - 1) * WALL_SLOTS + x - 1])
    ) {
      nIdx = idx - 1;
      if (DIST[nIdx] === -1) {
        DIST[nIdx] = nextDistance;
        QUEUE[tail++] = nIdx;
      }
    }

    // 右
    if (
      x < WALL_SLOTS &&
      (y === WALL_SLOTS || !s.vWalls[y * WALL_SLOTS + x]) &&
      (y === 0 || !s.vWalls[(y - 1) * WALL_SLOTS + x])
    ) {
      nIdx = idx + 1;
      if (DIST[nIdx] === -1) {
        DIST[nIdx] = nextDistance;
        QUEUE[tail++] = nIdx;
      }
    }
  }
  return -1;
}

// 各マスから p のゴール行までの最短歩数（壁のみ考慮。到達不能は -1）。
// 結果の配列は次の呼び出しで上書きされるので、すぐに読むこと
const GOAL_DIST = new Int16Array(SIZE * SIZE);

export function goalDistances(s: GameState, p: Player): Int16Array {
  const goal = goalRow(p);
  GOAL_DIST.fill(-1);
  let head = 0;
  let tail = 0;
  for (let x = 0; x < SIZE; x++) {
    const idx = goal * SIZE + x;
    GOAL_DIST[idx] = 0;
    QUEUE[tail++] = idx;
  }
  // 壁は両方向をふさぐので、ゴール側から逆にたどっても同じ距離になる
  while (head < tail) {
    const idx = QUEUE[head++];
    const x = idx % SIZE;
    const y = (idx - x) / SIZE;
    for (const [dx, dy] of DIRS) {
      if (!canStep(s, x, y, dx, dy)) continue;
      const nIdx = (y + dy) * SIZE + (x + dx);
      if (GOAL_DIST[nIdx] !== -1) continue;
      GOAL_DIST[nIdx] = GOAL_DIST[idx] + 1;
      QUEUE[tail++] = nIdx;
    }
  }
  return GOAL_DIST;
}

// ゴール行までの最短経路（始点を含む）。到達不能なら null。
export function shortestPath(s: GameState, p: Player): Pos[] | null {
  const start = s.pawns[p];
  const goal = goalRow(p);
  if (start.y === goal) return [start];
  const prev = new Int16Array(SIZE * SIZE).fill(-1);
  const startIdx = start.y * SIZE + start.x;
  prev[startIdx] = startIdx;
  const queue = new Int16Array(SIZE * SIZE);
  let head = 0;
  let tail = 0;
  queue[tail++] = startIdx;
  // ゴール方向を先に探索すると経路が自然になる
  const order = p === 0 ? [DIRS[0], DIRS[2], DIRS[3], DIRS[1]] : [DIRS[1], DIRS[2], DIRS[3], DIRS[0]];
  while (head < tail) {
    const idx = queue[head++];
    const x = idx % SIZE;
    const y = (idx - x) / SIZE;
    for (const [dx, dy] of order) {
      if (!canStep(s, x, y, dx, dy)) continue;
      const nIdx = (y + dy) * SIZE + (x + dx);
      if (prev[nIdx] !== -1) continue;
      prev[nIdx] = idx;
      if (y + dy === goal) {
        const path: Pos[] = [];
        for (let i = nIdx; ; i = prev[i]) {
          path.push({ x: i % SIZE, y: Math.floor(i / SIZE) });
          if (i === startIdx) break;
        }
        return path.reverse();
      }
      queue[tail++] = nIdx;
    }
  }
  return null;
}

// 既存の壁と重ならない・交差しないか（経路の封鎖チェックは含まない）
export function wallFits(s: GameState, x: number, y: number, o: Orientation): boolean {
  if (!inSlots(x, y)) return false;
  if (o === 'h') {
    return (
      !wallAt(s.hWalls, x, y) &&
      !wallAt(s.hWalls, x - 1, y) &&
      !wallAt(s.hWalls, x + 1, y) &&
      !wallAt(s.vWalls, x, y)
    );
  }
  return (
    !wallAt(s.vWalls, x, y) &&
    !wallAt(s.vWalls, x, y - 1) &&
    !wallAt(s.vWalls, x, y + 1) &&
    !wallAt(s.hWalls, x, y)
  );
}

export function isLegalWall(s: GameState, x: number, y: number, o: Orientation): boolean {
  if (isOver(s) || s.wallsLeft[s.turn] <= 0) return false;
  if (!wallFits(s, x, y, o)) return false;
  const walls = o === 'h' ? s.hWalls : s.vWalls;
  const i = y * WALL_SLOTS + x;
  walls[i] = 1;
  const ok = shortestDistance(s, 0) >= 0 && shortestDistance(s, 1) >= 0;
  walls[i] = 0;
  return ok;
}

export function legalWalls(s: GameState): Move[] {
  const result: Move[] = [];
  if (s.wallsLeft[s.turn] <= 0) return result;
  for (let y = 0; y < WALL_SLOTS; y++) {
    for (let x = 0; x < WALL_SLOTS; x++) {
      for (const o of ['h', 'v'] as const) {
        if (isLegalWall(s, x, y, o)) result.push({ type: 'wall', x, y, o });
      }
    }
  }
  return result;
}

export function legalMoves(s: GameState): Move[] {
  if (isOver(s)) return [];
  const pawns: Move[] = pawnMoves(s).map((p) => ({ type: 'pawn', x: p.x, y: p.y }));
  return pawns.concat(legalWalls(s));
}

export function isLegal(s: GameState, m: Move): boolean {
  if (isOver(s)) return false;
  if (m.type === 'wall') return isLegalWall(s, m.x, m.y, m.o);
  return pawnMoves(s).some((p) => p.x === m.x && p.y === m.y);
}

// 合法性はチェックしない（呼び出し側の責務）。新しい状態を返す。
export function applyMove(s: GameState, m: Move): GameState {
  const n = cloneState(s);
  applyMoveInPlace(n, m);
  return n;
}

export function applyMoveInPlace(s: GameState, m: Move): void {
  const p = s.turn;
  if (m.type === 'pawn') {
    s.pawns[p] = { x: m.x, y: m.y };
    if (m.y === goalRow(p)) s.winner = p;
  } else {
    (m.o === 'h' ? s.hWalls : s.vWalls)[m.y * WALL_SLOTS + m.x] = 1;
    s.wallsLeft[p]--;
  }
  s.turn = (1 - p) as Player;
  s.ply++;
  if (s.winner === null && s.ply >= MAX_PLIES) s.draw = true;
}

// 探索用: 盤面を複製せずにその場で指し、unmakeMove で元に戻す
export interface Undo {
  pawn: Pos;
  winner: Player | null;
  draw: boolean;
}

export function makeMove(s: GameState, m: Move): Undo {
  const undo: Undo = { pawn: s.pawns[s.turn], winner: s.winner, draw: s.draw };
  applyMoveInPlace(s, m);
  return undo;
}

export function unmakeMove(s: GameState, m: Move, undo: Undo): void {
  const p = (1 - s.turn) as Player;
  s.turn = p;
  s.ply--;
  s.winner = undo.winner;
  s.draw = undo.draw;
  if (m.type === 'pawn') {
    s.pawns[p] = undo.pawn;
  } else {
    (m.o === 'h' ? s.hWalls : s.vWalls)[m.y * WALL_SLOTS + m.x] = 0;
    s.wallsLeft[p]++;
  }
}

// 棋譜用の表記。列 a-i（左から）、行 1-9（下から）。壁はアンカー左上のマス + h/v（例: e3h）。
export function moveToString(m: Move): string {
  const col = String.fromCharCode(97 + m.x);
  if (m.type === 'pawn') return `${col}${SIZE - m.y}`;
  return `${col}${SIZE - 1 - m.y}${m.o}`;
}

export function moveFromString(str: string): Move | null {
  const match = /^([a-i])([1-9])([hv])?$/.exec(str.trim());
  if (!match) return null;
  const x = match[1].charCodeAt(0) - 97;
  const row = Number(match[2]);
  if (match[3]) {
    const y = SIZE - 1 - row;
    if (!inSlots(x, y)) return null;
    return { type: 'wall', x, y, o: match[3] as Orientation };
  }
  return { type: 'pawn', x, y: SIZE - row };
}
