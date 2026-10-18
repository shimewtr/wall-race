// 自己対戦の 1 局を実行する。Node のワーカーからも呼ばれるので DOM・fs に依存しない。

import { type SearchOptions, type Weights, recentCells, search } from '../engine/ai';
import { type GameState, type Player, applyMove, initialState, isOver } from '../engine/game';

export interface MatchConfig {
  // [先手の重み, 後手の重み]
  weights: [Partial<Weights>, Partial<Weights>];
  depth: number;
  // 最善手から margin 点以内の手からランダムに選ぶ（同じ棋譜の繰り返しを避ける）
  margin: number;
  // 序盤のこの手数だけは、最善から OPENING_MARGIN 点以内の手からランダムに選ぶ（展開をばらけさせる）
  openingPlies: number;
  seed: number;
  // 先手・後手それぞれの追加の探索オプション（実験で探索方法を比べる用）
  options?: [Partial<SearchOptions>, Partial<SearchOptions>];
}

// 序盤にばらけさせる手数と幅（1 歩 = 10 点。最善から 1 歩分以内の手なら選ぶ）。
// これが無いと自己対戦の序盤が毎回同じになり、統計や勝率が特定の展開に偏る
export const OPENING_PLIES = 6;
export const OPENING_MARGIN = 10;

// その手数での選び方の幅
export const marginAt = (ply: number, margin: number, openingPlies: number) =>
  ply < openingPlies ? Math.max(margin, OPENING_MARGIN) : margin;

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
    const { move } = search(s, {
      ...cfg.options?.[s.turn],
      maxDepth: cfg.depth,
      weights: cfg.weights[s.turn],
      margin: marginAt(s.ply, cfg.margin, cfg.openingPlies),
      rng,
      recent: recentCells(history, s.turn),
    });
    s = applyMove(s, move);
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
  const second = playMatch({
    ...opts,
    weights: [b, a],
    options: opts.options ? [opts.options[1], opts.options[0]] : undefined,
  });
  const score = (w: Player | null, aIs: Player) => (w === null ? 0.5 : w === aIs ? 1 : 0);
  return score(first, 0) + score(second, 1);
}
