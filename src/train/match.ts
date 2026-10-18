// 自己対戦の 1 局を実行する。Node のワーカーからも呼ばれるので DOM・fs に依存しない。

import { type Weights, recentCells, search } from '../engine/ai';
import { type GameState, type Player, applyMove, initialState, isOver, pawnMoves } from '../engine/game';

export interface MatchConfig {
  // [先手の重み, 後手の重み]
  weights: [Partial<Weights>, Partial<Weights>];
  depth: number;
  // 最善手から margin 点以内の手からランダムに選ぶ（同じ棋譜の繰り返しを避ける）
  margin: number;
  // 序盤に完全ランダムなコマ移動をする手数
  openingPlies: number;
  seed: number;
}

// 再現性のための軽量な乱数（mulberry32）
export function rngFrom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 勝者（引き分けは null）
export function playMatch(cfg: MatchConfig): Player | null {
  const rng = rngFrom(cfg.seed);
  const history: GameState[] = [initialState()];
  let s = history[0];
  while (!isOver(s)) {
    if (s.ply < cfg.openingPlies) {
      const ps = pawnMoves(s);
      const p = ps[Math.floor(rng() * ps.length)];
      s = applyMove(s, { type: 'pawn', x: p.x, y: p.y });
    } else {
      const { move } = search(s, {
        maxDepth: cfg.depth,
        weights: cfg.weights[s.turn],
        margin: cfg.margin,
        rng,
        recent: recentCells(history, s.turn),
      });
      s = applyMove(s, move);
    }
    history.push(s);
  }
  return s.winner;
}

// A と B で先後を入れ替えて同じ序盤から 2 局指し、A の得点（0〜2、引き分けは 0.5）を返す
export function playPair(
  a: Partial<Weights>,
  b: Partial<Weights>,
  opts: Omit<MatchConfig, 'weights'>,
): number {
  const first = playMatch({ ...opts, weights: [a, b] });
  const second = playMatch({ ...opts, weights: [b, a] });
  const score = (w: Player | null, aIs: Player) => (w === null ? 0.5 : w === aIs ? 1 : 0);
  return score(first, 0) + score(second, 1);
}
