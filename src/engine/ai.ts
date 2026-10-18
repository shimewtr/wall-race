// 探索 AI（negamax + αβ枝刈り + 反復深化）。
// 評価関数は重みベクトルで表現しているので、自己対戦でこの重みを調整して強くしていく想定。

import {
  type GameState,
  type Move,
  type Player,
  type Pos,
  WALL_SLOTS,
  applyMove,
  goalRow,
  isLegalWall,
  isOver,
  pawnMoves,
  shortestDistance,
  shortestPath,
} from './game';

// 評価に使う特徴量。すべて「手番側から見て」大きいほど良い向きで定義する。
// 学習で重みが決まると、そのまま「何を重視すべきか」の人間向けの説明になる。
export const FEATURES = {
  pathDiff: '最短経路の差（相手の残り歩数 - 自分の残り歩数）',
  rowDiff: '壁を無視した前進の差（相手のゴールまでの段数 - 自分の段数）',
  wallDiff: '残り壁の差（自分 - 相手）',
  myWalls: '自分の残り壁の数',
  urgency: '終盤の 1 歩の重さ（1/(自分の歩数+1) - 1/(相手の歩数+1)）',
  wallsWhenClose: '相手がゴールに近いほど自分の壁に価値（自分の壁/(相手の歩数+1) - 相手の壁/(自分の歩数+1)）',
  center: '中央寄り（相手の中央からの横ずれ - 自分の横ずれ）',
  adjacent: '相手と隣接している（手番側は飛び越えのチャンス）',
  lockedRace: '壁が尽きて歩数勝負が確定している（相手の壁 0 で自分が先着できる +1、自分の壁 0 で相手が先着できる -1）',
} as const;

export type Feature = keyof typeof FEATURES;
export type Weights = Record<Feature, number>;
export const FEATURE_NAMES = Object.keys(FEATURES) as Feature[];

export const DEFAULT_WEIGHTS: Weights = {
  pathDiff: 10,
  rowDiff: 0,
  wallDiff: 2,
  myWalls: 0.5,
  urgency: 20,
  wallsWhenClose: 0,
  center: 0,
  adjacent: 0,
  lockedRace: 0,
};

// 古い保存データに無い特徴量はデフォルト値で補う
export const withDefaults = (w: Partial<Weights> = {}): Weights => ({ ...DEFAULT_WEIGHTS, ...w });

export interface SearchOptions {
  maxDepth: number;
  timeMs?: number;
  weights?: Partial<Weights>;
  // 評価値に加えるランダムノイズの幅。弱い手も選ばれうるので、わざと手加減するレベル向け
  noise?: number;
  // 最善手からこの点数差以内の手の中からランダムに選ぶ（1 歩 = 10 点）。
  // 強さを落とさずに毎回同じ棋譜になるのを避けるためのもの
  margin?: number;
  rng?: () => number;
  // 自分のコマが直近にいたマス。ここへ戻る手は少し減点して、行ったり来たりを防ぐ
  recent?: Pos[];
  // 戻る手の減点（1 歩 = 10 点。同点に近い手どうしの順位だけを変える大きさ）
  repeatPenalty?: number;
  // いま指す手の候補に、守りの壁やコマ周りの壁も含める（読みの途中は相手の経路をふさぐ壁だけ）
  wideRoot?: boolean;
}

export const REPEAT_PENALTY = 8;

// 対局の状態履歴から、player のコマが直近 n 手番でいたマスを返す（現在地は除く）
export function recentCells(history: GameState[], player: Player, n = 4): Pos[] {
  const cur = history[history.length - 1].pawns[player];
  const cells: Pos[] = [];
  for (let i = history.length - 2; i >= 0 && cells.length < n; i--) {
    if (history[i].turn !== player) continue;
    const p = history[i].pawns[player];
    if (p.x !== cur.x || p.y !== cur.y) cells.push(p);
  }
  return cells;
}

export interface SearchResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
}

const WIN = 100000;

class Timeout extends Error {}

export function features(s: GameState): Weights {
  const me = s.turn;
  const op = (1 - me) as Player;
  const myDist = shortestDistance(s, me);
  const opDist = shortestDistance(s, op);
  const my = s.pawns[me];
  const o = s.pawns[op];
  return {
    pathDiff: opDist - myDist,
    rowDiff: Math.abs(goalRow(op) - o.y) - Math.abs(goalRow(me) - my.y),
    wallDiff: s.wallsLeft[me] - s.wallsLeft[op],
    myWalls: s.wallsLeft[me],
    urgency: 1 / (myDist + 1) - 1 / (opDist + 1),
    wallsWhenClose: s.wallsLeft[me] / (opDist + 1) - s.wallsLeft[op] / (myDist + 1),
    center: Math.abs(o.x - 4) - Math.abs(my.x - 4),
    adjacent: Math.abs(my.x - o.x) + Math.abs(my.y - o.y) === 1 ? 1 : 0,
    // 手番側は同じ歩数なら先に着く。壁が無ければ相手は邪魔できない
    lockedRace:
      (s.wallsLeft[op] === 0 && myDist <= opDist ? 1 : 0) - (s.wallsLeft[me] === 0 && opDist < myDist ? 1 : 0),
  };
}

// 手番側から見た静的評価
export function evaluate(s: GameState, w: Weights = DEFAULT_WEIGHTS): number {
  if (s.winner !== null) return s.winner === s.turn ? WIN : -WIN;
  if (s.draw) return 0;
  const f = features(s);
  let score = 0;
  for (const k of FEATURE_NAMES) score += w[k] * f[k];
  return score;
}

// 探索する壁の候補。
// 基本は「相手の最短経路の一歩一歩をふさぐ壁」だけ（読みの途中はこれで十分かつ軽い）。
// wide のときは加えて、
//   - どちらかのコマに接する壁（足止め・回り込みの準備）
//   - 自分の最短経路に接していて、自分の経路は伸ばさない壁（相手に道をふさがれないための守りの壁）
// も候補にする。根（いま指す手）でだけ使う。
export function candidateWalls(s: GameState, wide = false): Move[] {
  if (s.wallsLeft[s.turn] <= 0) return [];
  const me = s.turn;
  const op = (1 - me) as Player;
  const path = shortestPath(s, op);
  if (!path) return [];
  const seen = new Set<number>();
  const result: Move[] = [];
  const add = (x: number, y: number, o: 'h' | 'v', keep: (m: Move) => boolean = () => true) => {
    if (x < 0 || y < 0 || x >= WALL_SLOTS || y >= WALL_SLOTS) return;
    const key = (o === 'h' ? 0 : 64) + y * WALL_SLOTS + x;
    if (seen.has(key)) return;
    seen.add(key);
    const m: Move = { type: 'wall', x, y, o };
    if (isLegalWall(s, x, y, o) && keep(m)) result.push(m);
  };
  // マス (cx, cy) の辺に接する壁のアンカー
  const around = (cx: number, cy: number, keep?: (m: Move) => boolean) => {
    for (const ay of [cy - 1, cy]) {
      for (const ax of [cx - 1, cx]) {
        add(ax, ay, 'h', keep);
        add(ax, ay, 'v', keep);
      }
    }
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
  if (!wide) return result;
  for (const p of s.pawns) around(p.x, p.y);
  const myPath = shortestPath(s, me);
  if (myPath) {
    const myDist = myPath.length - 1;
    const keepsMyPath = (m: Move) => shortestDistance(applyMove(s, m), me) === myDist;
    for (const c of myPath) around(c.x, c.y, keepsMyPath);
  }
  return result;
}

function orderedMoves(s: GameState, wide = false): Move[] {
  const me = s.turn;
  const pawns = pawnMoves(s).map((p) => {
    const m: Move = { type: 'pawn', x: p.x, y: p.y };
    return { m, d: shortestDistance(applyMove(s, m), me) };
  });
  pawns.sort((a, b) => a.d - b.d);
  return [...pawns.map((p) => p.m), ...candidateWalls(s, wide)];
}

export function search(s: GameState, opts: SearchOptions): SearchResult {
  const weights = withDefaults(opts.weights);
  const rng = opts.rng ?? Math.random;
  const noise = opts.noise ?? 0;
  const margin = opts.margin ?? 0;
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

  const rootMoves = orderedMoves(s, opts.wideRoot ?? true);
  const recent = opts.recent ?? [];
  const repeatPenalty = opts.repeatPenalty ?? REPEAT_PENALTY;
  const penalty = (m: Move) =>
    m.type === 'pawn' && recent.some((p) => p.x === m.x && p.y === m.y) ? repeatPenalty : 0;
  if (rootMoves.length === 0) throw new Error('no legal moves');
  let result: SearchResult = { move: rootMoves[0], score: 0, depth: 0, nodes: 0 };

  for (let depth = 1; depth <= opts.maxDepth; depth++) {
    try {
      let bestScore = -Infinity;
      let bestMove = rootMoves[0];
      let alpha = -Infinity;
      const scored: Array<{ m: Move; v: number }> = [];
      for (const m of rootMoves) {
        // 最善手から margin 以内の手も正確な値が出るよう、窓を margin 分広げて読む
        const v =
          -negamax(applyMove(s, m), depth - 1, -Infinity, -(alpha - margin), 1) -
          penalty(m) +
          (noise ? (rng() - 0.5) * noise : 0);
        scored.push({ m, v });
        if (v > bestScore) {
          bestScore = v;
          bestMove = m;
        }
        if (v > alpha) alpha = v;
      }
      if (margin > 0) {
        const near = scored.filter((x) => x.v >= bestScore - margin);
        bestMove = near[Math.floor(rng() * near.length)].m;
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
  hard: { maxDepth: 3, timeMs: 5000, margin: 1 },
} satisfies Record<string, SearchOptions>;

export type Level = keyof typeof LEVELS;
