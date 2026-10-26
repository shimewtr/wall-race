// 長期停滞時に、現在位置の継続・学習率の再加熱・チャンピオンからの再出発を
// 同じ計算量と乱数系列で比較する。training/ は変更しない。
// 重い自己対戦を含むため GitHub Actions の Experiment で実行する。

import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { DEFAULT_WEIGHTS, FEATURE_NAMES, type Feature, type Weights, withDefaults } from '../../src/engine/ai';
import { OPENING_PLIES, rngFrom } from '../../src/train/match';
import { PairPool, type PairJob } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    iterations: { type: 'string', default: '12' },
    pairs: { type: 'string', default: '32' },
    seeds: { type: 'string', default: '3' },
    'eval-games': { type: 'string', default: '200' },
    strategies: { type: 'string', default: 'continue,reheat,restart' },
    seed: { type: 'string', default: '103000000' },
  },
});

const iterations = Number(args.iterations);
const pairsPerIteration = Number(args.pairs);
const seedCount = Number(args.seeds);
const evalGames = Number(args['eval-games']);
const baseSeed = Number(args.seed);
if (evalGames % 2 !== 0) throw new Error('eval-games must be even');

const root = join(import.meta.dirname, '..', '..');
const trainingDir = join(root, 'training');
const championFile = JSON.parse(readFileSync(join(trainingDir, 'champion.json'), 'utf8')) as {
  generation: number;
  weights: Partial<Weights>;
};
const stateFile = JSON.parse(readFileSync(join(trainingDir, 'state.json'), 'utf8')) as {
  theta: Partial<Weights>;
  stepIteration?: number;
};
const champion = withDefaults(championFile.weights);
const current = withDefaults(stateFile.theta);
const initial = withDefaults(DEFAULT_WEIGHTS);
const league = [
  initial,
  ...readdirSync(join(trainingDir, 'champions'))
    .filter((name) => name.endsWith('.json'))
    .sort()
    .map((name) => withDefaults(JSON.parse(readFileSync(join(trainingDir, 'champions', name), 'utf8')).weights)),
];

const fixed = new Set<Feature>(['pathDiff']);
const tuned = FEATURE_NAMES.filter((feature) => !fixed.has(feature));
const perturb: Record<Feature, number> = {
  pathDiff: 0,
  rowDiff: 1,
  wallDiff: 1,
  urgency: 4,
  wallsWhenClose: 2,
  center: 1,
  adjacent: 2,
  lockedRace: 5,
  wallThreat: 2,
  netWallThreat: 2,
  wallChain: 1,
  routeDiversity: 2,
};
const perturbMultiplier = 0.5;
const learningRate = (step: number) => 0.25 / (1 + step / 5000) ** 0.6;
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);

type StrategyName = 'continue' | 'reheat' | 'restart';
interface Strategy {
  name: StrategyName;
  start: Weights;
  stepIteration: number;
}
const allStrategies: Strategy[] = [
  { name: 'continue', start: current, stepIteration: stateFile.stepIteration ?? 0 },
  { name: 'reheat', start: current, stepIteration: 0 },
  { name: 'restart', start: champion, stepIteration: 0 },
];
const requested = new Set(args.strategies.split(','));
const strategies = allStrategies.filter((strategy) => requested.has(strategy.name));
if (strategies.length !== requested.size || strategies.length === 0) {
  throw new Error('strategies は continue,reheat,restart からカンマ区切りで指定してください');
}

async function train(pool: PairPool, strategy: Strategy, replicate: number): Promise<Weights> {
  const theta = { ...strategy.start };
  const seed = baseSeed + replicate * 10_000_000;
  for (let iteration = 0; iteration < iterations; iteration++) {
    const directionRng = rngFrom(seed + iteration);
    const delta = Object.fromEntries(tuned.map((feature) => [feature, directionRng() < 0.5 ? -1 : 1])) as Record<Feature, number>;
    const plus = { ...theta };
    const minus = { ...theta };
    for (const feature of tuned) {
      const amount = perturb[feature] * perturbMultiplier;
      plus[feature] += amount * delta[feature];
      minus[feature] -= amount * delta[feature];
    }
    const matchRng = rngFrom(seed + 1_000_000 + iteration);
    const matches = Array.from({ length: pairsPerIteration }, (_, index) => ({
      opponent: matchRng() < 0.5 ? champion : league[Math.floor(matchRng() * league.length)],
      opts: {
        depth: 3,
        margin: 1,
        openingPlies: OPENING_PLIES,
        seed: seed + 2_000_000 + iteration * pairsPerIteration + index,
      },
    }));
    const [plusScores, minusScores] = await Promise.all([plus, minus].map((weights) =>
      pool.runAll(matches.map((match) => ({ a: weights, b: match.opponent, opts: match.opts }))),
    ));
    const reward = (sum(plusScores) - sum(minusScores)) / (pairsPerIteration * 2);
    const rate = learningRate(strategy.stepIteration + iteration);
    for (const feature of tuned) {
      theta[feature] += rate * perturb[feature] * perturbMultiplier * reward * delta[feature];
    }
  }
  return theta;
}

async function compare(
  pool: PairPool,
  candidate: Weights,
  opponent: Weights | 'league',
  replicate: number,
  offset: number,
): Promise<number> {
  const pairCount = evalGames / 2;
  const jobs: PairJob[] = Array.from({ length: pairCount }, (_, index) => ({
    a: candidate,
    b: opponent === 'league' ? league[index % league.length] : opponent,
    opts: {
      depth: 3,
      margin: 1,
      openingPlies: OPENING_PLIES,
      seed: baseSeed + 50_000_000 + replicate * 1_000_000 + offset + index,
    },
  }));
  const scores = await pool.runAll(jobs);
  return sum(scores) / (pairCount * 2);
}

interface ResultRow {
  strategy: StrategyName;
  replicate: number;
  champion: number;
  initial: number;
  league: number;
  movementFromChampion: number;
  weights: Weights;
}
const pool = new PairPool();
const rows: ResultRow[] = [];
try {
  for (let replicate = 0; replicate < seedCount; replicate++) {
    for (const strategy of strategies) {
      const weights = await train(pool, strategy, replicate);
      const championScore = await compare(pool, weights, champion, replicate, 100_000);
      const initialScore = await compare(pool, weights, initial, replicate, 200_000);
      const leagueScore = await compare(pool, weights, 'league', replicate, 300_000);
      const movementFromChampion = Math.sqrt(sum(tuned.map((feature) => (weights[feature] - champion[feature]) ** 2)));
      rows.push({ strategy: strategy.name, replicate, champion: championScore, initial: initialScore, league: leagueScore, movementFromChampion, weights });
    }
  }
} finally {
  await pool.close();
}

const average = (values: number[]) => sum(values) / values.length;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const summary = Object.fromEntries(strategies.map(({ name }) => {
  const selected = rows.filter((row) => row.strategy === name);
  return [name, {
    champion: average(selected.map((row) => row.champion)),
    initial: average(selected.map((row) => row.initial)),
    league: average(selected.map((row) => row.league)),
    movementFromChampion: average(selected.map((row) => row.movementFromChampion)),
    championRange: [Math.min(...selected.map((row) => row.champion)), Math.max(...selected.map((row) => row.champion))],
  }];
})) as Partial<Record<StrategyName, {
  champion: number;
  initial: number;
  league: number;
  movementFromChampion: number;
  championRange: [number, number];
}>>;

console.log(`# 第${championFile.generation}世代の停滞リスタート予備実験`);
console.log(`各方式 ${seedCount} seed × ${iterations}反復 × ${pairsPerIteration}組、評価は各候補・各指標${evalGames}局`);
console.log(`現在の stepIteration=${stateFile.stepIteration ?? 0}、学習率=${learningRate(stateFile.stepIteration ?? 0).toFixed(4)}、再加熱後=${learningRate(0).toFixed(4)}`);
console.log('\n| 方式 | 対チャンピオン | seed範囲 | 対初期 | 歴代混合 | チャンピオンからの移動量 |');
console.log('| --- | ---: | ---: | ---: | ---: | ---: |');
for (const { name } of strategies) {
  const item = summary[name];
  if (!item) continue;
  console.log(`| ${name} | ${percent(item.champion)} | ${percent(item.championRange[0])}〜${percent(item.championRange[1])} | ${percent(item.initial)} | ${percent(item.league)} | ${item.movementFromChampion.toFixed(3)} |`);
}
console.log('\nseed別:');
for (const row of rows) {
  console.log(`  ${row.strategy} #${row.replicate + 1}: champion ${percent(row.champion)} / initial ${percent(row.initial)} / league ${percent(row.league)}`);
}
console.log('\n判定: 3指標でcontinueを悪化させず、複数seedで方向が一致する方式だけを400局以上の再確認へ進める。');
writeFileSync('experiment-output.json', `${JSON.stringify({
  generation: championFile.generation,
  source: { iterations, pairsPerIteration, seedCount, evalGames, baseSeed, currentStepIteration: stateFile.stepIteration ?? 0 },
  summary,
  rows,
}, null, 2)}\n`);
