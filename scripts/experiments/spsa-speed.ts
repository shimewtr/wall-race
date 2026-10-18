// 学習（SPSA）の 1 回分の対局が、ベンチマークよりなぜ遅いかを切り分ける。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。

import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { type Weights, withDefaults } from '../../src/engine/ai';
import { OPENING_PLIES } from '../../src/train/match';
import { PairPool } from '../lib/pool';

const DIR = join(import.meta.dirname, '..', '..', 'training');
const champion = withDefaults(JSON.parse(readFileSync(join(DIR, 'champion.json'), 'utf8')).weights);
const theta = existsSync(join(DIR, 'state.json')) ? withDefaults(JSON.parse(readFileSync(join(DIR, 'state.json'), 'utf8')).theta) : champion;
const gen1 = withDefaults(JSON.parse(readFileSync(join(DIR, 'champions', readdirSync(join(DIR, 'champions')).sort()[0]), 'utf8')).weights);

const pool = new PairPool();
const pairs = 32;
async function measure(label: string, a: Weights, b: Weights, openingPlies: number) {
  const t = Date.now();
  await pool.runAll(Array.from({ length: pairs }, (_, i) => ({ a, b, opts: { depth: 3, margin: 1, openingPlies, seed: 50_000 + i } })));
  console.log(`${label.padEnd(36)} ${((Date.now() - t) / 1000).toFixed(1)} 秒 / ${pairs} 組`);
}
await measure('チャンピオン同士・序盤ランダムなし', champion, champion, 0);
await measure(`チャンピオン同士・序盤 ${OPENING_PLIES} 手ランダム`, champion, champion, OPENING_PLIES);
await measure('学習中の重み 対 チャンピオン・序盤ランダム', theta, champion, OPENING_PLIES);
await measure('学習中の重み 対 第 1 世代・序盤ランダム', theta, gen1, OPENING_PLIES);
await pool.close();
