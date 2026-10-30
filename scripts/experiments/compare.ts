// 2 つの AI 設定（重み・探索オプション）を対戦させて勝率と 1 手あたりの思考時間を比べる。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/compare.ts --a '{"wideRoot":true}' --b '{"wideRoot":false}' --pairs 100
//     --wa / --wb は重み（champion / depth4-teacher / training / initial / JSON / champion+JSON）。省略時はどちらも champion

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { type SearchOptions, type Weights, search, withDefaults } from '../../src/engine/ai';
import { type GameState, applyMove, initialState, isOver } from '../../src/engine/game';
import { OPENING_PLIES, rngFrom } from '../../src/train/match';
import { PairPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    a: { type: 'string', default: '{}' },
    b: { type: 'string', default: '{}' },
    wa: { type: 'string', default: 'champion' },
    wb: { type: 'string', default: 'champion' },
    depth: { type: 'string', default: '3' },
    pairs: { type: 'string', default: '100' },
    seed: { type: 'string', default: '7000000' },
  },
});

const CHAMPION = join(import.meta.dirname, '..', '..', 'training', 'champion.json');
const DEPTH4_TEACHER = join(import.meta.dirname, '..', '..', 'training', 'depth4-teacher.json');
const STATE = join(import.meta.dirname, '..', '..', 'training', 'state.json');
// champion / depth4-teacher / training / initial / JSON / champion+JSON（チャンピオンの重みの一部だけ変える）
function weights(spec: string): Weights {
  if (spec.startsWith('champion+')) return { ...weights('champion'), ...JSON.parse(spec.slice('champion+'.length)) };
  if (spec === 'initial') return withDefaults();
  if (spec === 'depth4-teacher') return withDefaults(JSON.parse(readFileSync(DEPTH4_TEACHER, 'utf8')).weights);
  if (spec === 'training') {
    return existsSync(STATE) ? withDefaults(JSON.parse(readFileSync(STATE, 'utf8')).theta) : withDefaults();
  }
  if (spec === 'champion') {
    return existsSync(CHAMPION) ? withDefaults(JSON.parse(readFileSync(CHAMPION, 'utf8')).weights) : withDefaults();
  }
  return withDefaults(JSON.parse(spec));
}

const depth = Number(args.depth);
const pairs = Number(args.pairs);
const seed = Number(args.seed);
const optA: Partial<SearchOptions> = JSON.parse(args.a!);
const optB: Partial<SearchOptions> = JSON.parse(args.b!);
const wA = weights(args.wa!);
const wB = weights(args.wb!);
console.log(`A: weights=${args.wa} options=${args.a}`);
console.log(`B: weights=${args.wb} options=${args.b}`);
console.log(`depth=${depth} pairs=${pairs} seed=${seed}\n`);

// 勝率（全コアで並列）
const pool = new PairPool();
const t0 = Date.now();
const scores = await pool.runAll(
  Array.from({ length: pairs }, (_, i) => ({
    a: wA,
    b: wB,
    opts: { depth, margin: 1, openingPlies: OPENING_PLIES, seed: seed + i, options: [optA, optB] as [typeof optA, typeof optB] },
  })),
);
await pool.close();
const total = scores.reduce((x, y) => x + y, 0);
const rate = total / (pairs * 2);
// 95% 信頼区間の目安（引き分けはほぼ無いので二項分布で近似）
const se = Math.sqrt((rate * (1 - rate)) / (pairs * 2));
console.log(`A の勝率: ${(rate * 100).toFixed(1)}% ±${(1.96 * se * 100).toFixed(1)}（${pairs * 2} 局, ${((Date.now() - t0) / 1000).toFixed(0)} 秒）`);

// 1 手あたりの思考時間（1 コアで、同じ局面を両方に考えさせる）
const positions: GameState[] = [];
for (let g = 0; g < 4; g++) {
  const rng = rngFrom(9_000 + g);
  let s = initialState();
  while (!isOver(s)) {
    positions.push(s);
    s = applyMove(s, search(s, { maxDepth: 2, weights: wA, margin: 1, rng }).move);
  }
}
const sample = positions.filter((_, i) => i % 3 === 0).slice(0, 60);
const time = (opts: Partial<SearchOptions>, w: Weights) => {
  const ms: number[] = [];
  for (const s of sample) {
    const t = performance.now();
    search(s, { maxDepth: depth, ...opts, weights: w });
    ms.push(performance.now() - t);
  }
  ms.sort((x, y) => x - y);
  return `平均 ${(ms.reduce((x, y) => x + y, 0) / ms.length).toFixed(0)}ms / 最大 ${ms[ms.length - 1].toFixed(0)}ms`;
};
console.log(`思考時間（${sample.length} 局面）: A ${time(optA, wA)}、B ${time(optB, wB)}`);
