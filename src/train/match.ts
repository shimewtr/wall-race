// 自己対戦の 1 局を実行する。Node のワーカーからも呼ばれるので DOM・fs に依存しない。

import { type SearchOptions, type Weights, recentCells, search } from '../engine/ai';
import { type GameState, type Player, applyMove, cloneState, initialState, isOver } from '../engine/game';

export interface MatchConfig {
  // [先手の重み, 後手の重み]
  weights: [Partial<Weights>, Partial<Weights>];
  depth: number;
  // 最善手から margin 点以内の手からランダムに選ぶ（同じ棋譜の繰り返しを避ける）
  margin: number;
  // 序盤のこの手数だけは、評価の上位 OPENING_TOP 手からランダムに選ぶ（展開をばらけさせる）
  openingPlies: number;
  seed: number;
  // 中間局面から始める実験用。末尾を開始局面とし、直前の状態は往復防止にも使う。
  // 省略時は通常どおり初期局面から始める。
  startHistory?: GameState[];
  // 先手・後手それぞれの追加の探索オプション（実験で探索方法を比べる用）
  options?: [Partial<SearchOptions>, Partial<SearchOptions>];
}

// 序盤にばらけさせる手数と、そのあいだ選ぶ候補の数（評価の上位 OPENING_TOP 手からランダム）。
// これが無いと自己対戦の序盤が毎回同じになり、統計や勝率が特定の展開に偏る。
// 「最善から N 点以内」で選ぶ方式は、序盤は前進と他の手の差が大きく、結局毎回同じ手になったので使わない
export const OPENING_PLIES = 6;
export const OPENING_TOP = 3;

// その手数で上位何手から選ぶか
export const pickAt = (ply: number, openingPlies: number) => (ply < openingPlies ? OPENING_TOP : 1);

// 序盤にばらけさせる間の読みの深さ。上位から選ぶ間は根で枝刈りできず重いので浅く読む
// （3 手読みのままだと自己対戦全体の時間が約 2 倍になった。ばらけさせるのが目的なので厳密さは要らない）
export const OPENING_DEPTH = 2;
export const depthAt = (ply: number, depth: number, openingPlies: number) =>
  ply < openingPlies ? Math.min(depth, OPENING_DEPTH) : depth;

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
  if (cfg.startHistory?.length === 0) throw new Error('startHistory must not be empty');
  const history: GameState[] = cfg.startHistory?.map(cloneState) ?? [initialState()];
  let s = history.at(-1)!;
  while (!isOver(s)) {
    const { move } = search(s, {
      ...cfg.options?.[s.turn],
      maxDepth: depthAt(s.ply, cfg.options?.[s.turn]?.maxDepth ?? cfg.depth, cfg.openingPlies),
      weights: cfg.weights[s.turn],
      margin: cfg.margin,
      pickFromTop: pickAt(s.ply, cfg.openingPlies),
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
