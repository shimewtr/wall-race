// 探索 AI（negamax + αβ枝刈り + 反復深化）。
// 評価関数は重みベクトルで表現しているので、自己対戦でこの重みを調整して強くしていく想定。

import {
  type GameState,
  type Move,
  type Player,
  WALL_SLOTS,
  applyMove,
  isLegalWall,
  isOver,
  pawnMoves,
  shortestDistance,
  shortestPath,
} from './game';

export interface Weights {
  // (相手の最短距離 - 自分の最短距離)
  pathDiff: number;
  // (自分の残り壁 - 相手の残り壁)
  wallDiff: number;
  // 自分の残り壁の絶対量（壁を温存する価値）
  myWalls: number;
  // ゴール間近の重み: (1/(自分の距離+1) - 1/(相手の距離+1))。終盤ほど 1 歩の価値が上がる
  urgency: number;
}

export const DEFAULT_WEIGHTS: Weights = {
  pathDiff: 10,
  wallDiff: 2,
  myWalls: 0.5,
  urgency: 20,
};

export interface SearchOptions {
  maxDepth: number;
  timeMs?: number;
  weights?: Weights;
  // 評価値に加えるランダムノイズの幅（自己対戦で同じ棋譜ばかりにならないように）
  noise?: number;
  rng?: () => number;
}

export interface SearchResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
}

const WIN = 100000;

class Timeout extends Error {}

// 手番側から見た静的評価
export function evaluate(s: GameState, w: Weights = DEFAULT_WEIGHTS): number {
  const me = s.turn;
  const op = (1 - me) as Player;
  if (s.winner !== null) return s.winner === me ? WIN : -WIN;
  if (s.draw) return 0;
  const myDist = shortestDistance(s, me);
  const opDist = shortestDistance(s, op);
  return (
    w.pathDiff * (opDist - myDist) +
    w.wallDiff * (s.wallsLeft[me] - s.wallsLeft[op]) +
    w.myWalls * s.wallsLeft[me] +
    w.urgency * (1 / (myDist + 1) - 1 / (opDist + 1))
  );
}

// 探索する壁の候補: 相手の最短経路上の一歩一歩をふさぐ壁だけに絞る
export function candidateWalls(s: GameState): Move[] {
  if (s.wallsLeft[s.turn] <= 0) return [];
  const op = (1 - s.turn) as Player;
  const path = shortestPath(s, op);
  if (!path) return [];
  const seen = new Set<number>();
  const result: Move[] = [];
  const add = (x: number, y: number, o: 'h' | 'v') => {
    if (x < 0 || y < 0 || x >= WALL_SLOTS || y >= WALL_SLOTS) return;
    const key = (o === 'h' ? 0 : 64) + y * WALL_SLOTS + x;
    if (seen.has(key)) return;
    seen.add(key);
    if (isLegalWall(s, x, y, o)) result.push({ type: 'wall', x, y, o });
  };
  for (let i = 0; i + 1 < path.length; i++) {
    const a = path[i];
    const b = path[i + 1];
    if (a.x === b.x) {
      const y = Math.min(a.y, b.y);
      add(a.x, y, 'h');
      add(a.x - 1, y, 'h');
    } else {
      const x = Math.min(a.x, b.x);
      add(x, a.y, 'v');
      add(x, a.y - 1, 'v');
    }
  }
  return result;
}

function orderedMoves(s: GameState): Move[] {
  const me = s.turn;
  const pawns = pawnMoves(s).map((p) => {
    const m: Move = { type: 'pawn', x: p.x, y: p.y };
    return { m, d: shortestDistance(applyMove(s, m), me) };
  });
  pawns.sort((a, b) => a.d - b.d);
  return [...pawns.map((p) => p.m), ...candidateWalls(s)];
}

export function search(s: GameState, opts: SearchOptions): SearchResult {
  const weights = opts.weights ?? DEFAULT_WEIGHTS;
  const rng = opts.rng ?? Math.random;
  const noise = opts.noise ?? 0;
  const deadline = opts.timeMs ? Date.now() + opts.timeMs : Infinity;
  let nodes = 0;

  const negamax = (st: GameState, depth: number, alpha: number, beta: number, ply: number): number => {
    nodes++;
    if ((nodes & 255) === 0 && Date.now() > deadline) throw new Timeout();
    if (isOver(st)) {
      // 早く勝つ・遅く負けるほうを好むように ply を加味
      if (st.winner === null) return 0;
      return st.winner === st.turn ? WIN - ply : -WIN + ply;
    }
    if (depth === 0) return evaluate(st, weights);
    let best = -Infinity;
    for (const m of orderedMoves(st)) {
      const v = -negamax(applyMove(st, m), depth - 1, -beta, -alpha, ply + 1);
      if (v > best) best = v;
      if (v > alpha) alpha = v;
      if (alpha >= beta) break;
    }
    return best === -Infinity ? evaluate(st, weights) : best;
  };

  const rootMoves = orderedMoves(s);
  if (rootMoves.length === 0) throw new Error('no legal moves');
  let result: SearchResult = { move: rootMoves[0], score: 0, depth: 0, nodes: 0 };

  for (let depth = 1; depth <= opts.maxDepth; depth++) {
    try {
      let bestScore = -Infinity;
      let bestMove = rootMoves[0];
      let alpha = -Infinity;
      const scored: Array<{ m: Move; v: number }> = [];
      for (const m of rootMoves) {
        const v = -negamax(applyMove(s, m), depth - 1, -Infinity, -alpha, 1) + (noise ? (rng() - 0.5) * noise : 0);
        scored.push({ m, v });
        if (v > bestScore) {
          bestScore = v;
          bestMove = m;
        }
        if (v > alpha) alpha = v;
      }
      result = { move: bestMove, score: bestScore, depth, nodes };
      // 次の深さでは良かった手から読む
      scored.sort((a, b) => b.v - a.v);
      rootMoves.splice(0, rootMoves.length, ...scored.map((x) => x.m));
      if (Math.abs(bestScore) > WIN / 2) break;
    } catch (e) {
      if (e instanceof Timeout) break;
      throw e;
    }
  }
  result.nodes = nodes;
  return result;
}

export const LEVELS = {
  easy: { maxDepth: 1, noise: 8 },
  normal: { maxDepth: 2, noise: 2 },
  hard: { maxDepth: 4, timeMs: 3000 },
} satisfies Record<string, SearchOptions>;

export type Level = keyof typeof LEVELS;
