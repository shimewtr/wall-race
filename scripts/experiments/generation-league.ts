// 歴代チャンピオンを総当たりさせ、単純な世代順では見えない相性と非推移性を調べる。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。

import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { type Weights, withDefaults } from '../../src/engine/ai';
import { OPENING_PLIES } from '../../src/train/match';
import { PairPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    pairs: { type: 'string', default: '50' },
    depth: { type: 'string', default: '3' },
    seed: { type: 'string', default: '99000000' },
  },
});

interface Generation {
  generation: number;
  weights: Weights;
}

interface PairResult {
  a: number;
  b: number;
  games: number;
  scoreA: number;
}

const pairs = Number(args.pairs);
const depth = Number(args.depth);
const seed = Number(args.seed);
if (!Number.isInteger(pairs) || pairs <= 0) throw new Error('--pairs must be a positive integer');
if (!Number.isInteger(depth) || depth <= 0) throw new Error('--depth must be a positive integer');

const championsDir = join(import.meta.dirname, '..', '..', 'training', 'champions');
const generations: Generation[] = readdirSync(championsDir)
  .filter((name) => /^gen-\d+\.json$/.test(name))
  .map((name) => JSON.parse(readFileSync(join(championsDir, name), 'utf8')) as Generation)
  .map((entry) => ({ generation: entry.generation, weights: withDefaults(entry.weights) }))
  .sort((x, y) => x.generation - y.generation);

if (generations.length < 3) throw new Error('at least three champion generations are required');

const matchups = generations.flatMap((a, i) => generations.slice(i + 1).map((b) => ({ a, b })));
console.log(`# 世代リーグ（${generations.length}世代・${matchups.length}組）`);
console.log(`深さ${depth}、各組${pairs * 2}局、seed=${seed}`);

const pool = new PairPool();
const startedAt = Date.now();
const scores = await pool.runAll(
  matchups.flatMap(({ a, b }, matchupIndex) =>
    Array.from({ length: pairs }, (_, pairIndex) => ({
      a: a.weights,
      b: b.weights,
      opts: {
        depth,
        margin: 1,
        openingPlies: OPENING_PLIES,
        seed: seed + matchupIndex * pairs + pairIndex,
      },
    })),
  ),
);
await pool.close();

const results: PairResult[] = matchups.map(({ a, b }, matchupIndex) => {
  const slice = scores.slice(matchupIndex * pairs, (matchupIndex + 1) * pairs);
  return { a: a.generation, b: b.generation, games: pairs * 2, scoreA: slice.reduce((sum, value) => sum + value, 0) };
});

const rate = (result: PairResult, generation: number) => {
  const score = result.scoreA / result.games;
  return generation === result.a ? score : 1 - score;
};
const byPair = new Map(results.map((result) => [`${result.a}:${result.b}`, result]));
const versus = (a: number, b: number) => {
  if (a === b) return 0.5;
  const result = byPair.get(`${Math.min(a, b)}:${Math.max(a, b)}`)!;
  return rate(result, a);
};
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const halfWidth = (value: number, games: number) => 1.96 * Math.sqrt((value * (1 - value)) / games);

// Bradley–Terry の1次元強さを当てはめ、そこから大きく外れる組を「相性」の候補にする。
const ratings = new Map(generations.map(({ generation }) => [generation, 0]));
for (let iteration = 0; iteration < 2_000; iteration++) {
  const gradient = new Map(generations.map(({ generation }) => [generation, 0]));
  const gamesPerGeneration = new Map(generations.map(({ generation }) => [generation, 0]));
  for (const result of results) {
    const expected = 1 / (1 + Math.exp(-(ratings.get(result.a)! - ratings.get(result.b)!)));
    const error = result.scoreA / result.games - expected;
    gradient.set(result.a, gradient.get(result.a)! + result.games * error);
    gradient.set(result.b, gradient.get(result.b)! - result.games * error);
    gamesPerGeneration.set(result.a, gamesPerGeneration.get(result.a)! + result.games);
    gamesPerGeneration.set(result.b, gamesPerGeneration.get(result.b)! + result.games);
  }
  for (const { generation } of generations) {
    ratings.set(generation, ratings.get(generation)! + 0.5 * gradient.get(generation)! / gamesPerGeneration.get(generation)!);
  }
  const mean = [...ratings.values()].reduce((sum, value) => sum + value, 0) / ratings.size;
  for (const generation of ratings.keys()) ratings.set(generation, ratings.get(generation)! - mean);
}

const expected = (a: number, b: number) => 1 / (1 + Math.exp(-(ratings.get(a)! - ratings.get(b)!)));
const residuals = results
  .map((result) => ({
    a: result.a,
    b: result.b,
    observedA: result.scoreA / result.games,
    expectedA: expected(result.a, result.b),
    residual: result.scoreA / result.games - expected(result.a, result.b),
  }))
  .sort((x, y) => Math.abs(y.residual) - Math.abs(x.residual));
const residualRmse = Math.sqrt(residuals.reduce((sum, item) => sum + item.residual ** 2, 0) / residuals.length);

console.log('\n## 勝率表（行が列に勝つ率）');
console.log(['世代', ...generations.map((g) => `G${g.generation}`), '平均'].join('\t'));
const averages = new Map<number, number>();
for (const a of generations) {
  const rates = generations.filter((b) => b.generation !== a.generation).map((b) => versus(a.generation, b.generation));
  const average = rates.reduce((sum, value) => sum + value, 0) / rates.length;
  averages.set(a.generation, average);
  console.log([
    `G${a.generation}`,
    ...generations.map((b) => (a.generation === b.generation ? '—' : pct(versus(a.generation, b.generation)))),
    pct(average),
  ].join('\t'));
}

console.log('\n## 1次元強さ（Bradley–Terry、平均0 Elo）');
for (const { generation } of [...generations].sort((a, b) => ratings.get(b.generation)! - ratings.get(a.generation)!)) {
  const elo = ratings.get(generation)! * 400 / Math.log(10);
  console.log(`G${generation}: ${elo >= 0 ? '+' : ''}${elo.toFixed(0)} Elo、総当たり平均 ${pct(averages.get(generation)!)}`);
}

console.log(`\n## 1次元モデルから最も外れる相性（残差RMSE ${pct(residualRmse)}）`);
for (const item of residuals.slice(0, 8)) {
  console.log(`G${item.a} 対 G${item.b}: 実測 ${pct(item.observedA)}、期待 ${pct(item.expectedA)}、差 ${item.residual >= 0 ? '+' : ''}${pct(item.residual)}`);
}

const latest = generations.at(-1)!.generation;
console.log(`\n## 最新G${latest}の相手別成績`);
for (const opponent of generations.slice(0, -1).sort((a, b) => versus(latest, a.generation) - versus(latest, b.generation))) {
  const value = versus(latest, opponent.generation);
  console.log(`対G${opponent.generation}: ${pct(value)} ±${pct(halfWidth(value, pairs * 2))}`);
}

const upsets = results
  .map((result) => ({ older: result.a, newer: result.b, rate: result.scoreA / result.games, games: result.games }))
  .filter((item) => item.rate - halfWidth(item.rate, item.games) > 0.5)
  .sort((x, y) => y.rate - x.rate);
console.log('\n## 95%区間でも50%を超える旧世代の勝ち越し');
if (upsets.length === 0) console.log('なし');
for (const item of upsets) console.log(`G${item.older} 対 G${item.newer}: ${pct(item.rate)} ±${pct(halfWidth(item.rate, item.games))}`);

const cycles: Array<{ generations: [number, number, number]; rates: [number, number, number]; strength: number }> = [];
for (let i = 0; i < generations.length; i++) {
  for (let j = i + 1; j < generations.length; j++) {
    for (let k = j + 1; k < generations.length; k++) {
      const [a, b, c] = [generations[i].generation, generations[j].generation, generations[k].generation];
      for (const order of [[a, b, c], [a, c, b]] as const) {
        const rates: [number, number, number] = [versus(order[0], order[1]), versus(order[1], order[2]), versus(order[2], order[0])];
        const strength = Math.min(...rates) - 0.5;
        if (strength > 0) cycles.push({ generations: [...order], rates, strength });
      }
    }
  }
}
cycles.sort((x, y) => y.strength - x.strength);
console.log('\n## 観測上の循環（探索候補、上位5件）');
if (cycles.length === 0) console.log('なし');
for (const cycle of cycles.slice(0, 5)) {
  const [a, b, c] = cycle.generations;
  console.log(`G${a} ${pct(cycle.rates[0])} > G${b} ${pct(cycle.rates[1])} > G${c} ${pct(cycle.rates[2])} > G${a}（最弱辺 +${pct(cycle.strength)}）`);
}

const output = {
  generatedAt: new Date().toISOString(),
  depth,
  pairs,
  seed,
  elapsedSeconds: (Date.now() - startedAt) / 1_000,
  generations: generations.map(({ generation }) => generation),
  results,
  ratings: Object.fromEntries([...ratings].map(([generation, value]) => [generation, value * 400 / Math.log(10)])),
  residualRmse,
  residuals,
  upsets,
  cycles: cycles.slice(0, 20),
};
writeFileSync('experiment-output.json', `${JSON.stringify(output, null, 2)}\n`);
console.log(`\n${results.reduce((sum, result) => sum + result.games, 0)}局、${((Date.now() - startedAt) / 1_000).toFixed(0)}秒。詳細はexperiment-output.jsonへ保存。`);
