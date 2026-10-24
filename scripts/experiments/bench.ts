// 探索の速さを測る。高速化の前後で比べる用。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/bench.ts --depth 3

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { search, withDefaults, type SearchOptions } from '../../src/engine/ai';
import { type GameState, applyMove, initialState, isOver } from '../../src/engine/game';
import { rngFrom } from '../../src/train/match';
import { PairPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    depth: { type: 'string', default: '3' },
    pairs: { type: 'string', default: '40' },
    weights: { type: 'string', default: 'champion' },
    options: { type: 'string', default: '{}' },
  },
});

const CHAMPION = join(import.meta.dirname, '..', '..', 'training', 'champion.json');
const champion = withDefaults(existsSync(CHAMPION) ? JSON.parse(readFileSync(CHAMPION, 'utf8')).weights : {});
const weights = args.weights!.startsWith('champion+')
  ? { ...champion, ...JSON.parse(args.weights!.slice('champion+'.length)) }
  : args.weights === 'champion'
    ? champion
    : withDefaults(JSON.parse(args.weights!));
const depth = Number(args.depth);
const options: Partial<SearchOptions> = JSON.parse(args.options!);
console.log(`weights=${args.weights}`);
console.log(`options=${args.options}`);

// 1 コアで、決まった局面集合を探索する時間とノード数
const positions: GameState[] = [];
for (let g = 0; g < 6; g++) {
  const rng = rngFrom(21_000 + g);
  let s = initialState();
  while (!isOver(s)) {
    positions.push(s);
    s = applyMove(s, search(s, { ...options, maxDepth: 2, weights, margin: 10, rng }).move);
  }
}
const sample = positions.filter((_, i) => i % 4 === 0).slice(0, 80);
let nodes = 0;
const t0 = performance.now();
const moves: string[] = [];
for (const s of sample) {
  const r = search(s, { ...options, maxDepth: depth, weights });
  nodes += r.nodes;
  moves.push(JSON.stringify(r.move));
}
const ms = performance.now() - t0;
console.log(`1 コア: ${sample.length} 局面を ${depth} 手読み → 平均 ${(ms / sample.length).toFixed(1)}ms/手、${Math.round(nodes / (ms / 1000))} ノード/秒`);
// 高速化で結果が変わっていないかの確認用（同じ局面・同じ重みなら同じ手になるはず）
let hash = 0;
for (const c of moves.join('|')) hash = (hash * 31 + c.charCodeAt(0)) >>> 0;
console.log(`選んだ手のハッシュ: ${hash.toString(16)}`);

// 全コアでの対局の速さ
const pool = new PairPool();
const pairs = Number(args.pairs);
const t1 = Date.now();
await pool.runAll(
  Array.from({ length: pairs }, (_, i) => ({
    a: weights,
    b: weights,
    opts: {
      depth,
      margin: 1,
      openingPlies: 0,
      seed: i,
      options: [options, options] as [typeof options, typeof options],
    },
  })),
);
await pool.close();
console.log(`全 ${pool.size} コア: ${(pairs / ((Date.now() - t1) / 1000)).toFixed(2)} 組/秒（1 組 = 先後入れ替えの 2 局）`);
