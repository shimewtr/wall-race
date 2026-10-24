// 決定的局面から始める対局を0%・10%・20%混ぜ、同じ計算量の短期SPSA学習を比較する。
// 通常Trainとは独立したExperimentで、training/は変更しない。

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { DEFAULT_WEIGHTS, FEATURE_NAMES, recentCells, search, withDefaults, type Feature, type Weights } from '../../src/engine/ai';
import { type GameState, applyMove, cloneState, initialState, isOver, moveToString } from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom, type MatchConfig } from '../../src/train/match';
import { PairPool, type PairJob } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    iterations: { type: 'string', default: '8' },
    pairs: { type: 'string', default: '32' },
    'eval-games': { type: 'string', default: '200' },
    'archive-positions': { type: 'string', default: '600' },
    'archive-limit': { type: 'string', default: '60' },
    rates: { type: 'string', default: '0,0.1,0.2' },
    seed: { type: 'string', default: '99000000' },
  },
});

const iterations = Number(args.iterations);
const pairsPerIteration = Number(args.pairs);
const evalGames = Number(args['eval-games']);
const archivePositions = Number(args['archive-positions']);
const archiveLimit = Number(args['archive-limit']);
const rates = args.rates.split(',').map(Number);
const seed = Number(args.seed);
if (evalGames % 2 !== 0) throw new Error('eval-games must be even');

const root = join(import.meta.dirname, '..', '..');
const trainingDir = join(root, 'training');
const champion = JSON.parse(readFileSync(join(trainingDir, 'champion.json'), 'utf8')) as { generation: number; weights: Weights };
const league = [
  withDefaults(),
  ...readdirSync(join(trainingDir, 'champions'))
    .filter((name) => name.endsWith('.json'))
    .sort()
    .map((name) => withDefaults(JSON.parse(readFileSync(join(trainingDir, 'champions', name), 'utf8')).weights)),
];
const TUNED = FEATURE_NAMES.filter((feature) => feature !== 'pathDiff');
const PERTURB: Record<Feature, number> = {
  pathDiff: 0, rowDiff: 1, wallDiff: 1, urgency: 4, wallsWhenClose: 2, center: 1,
  adjacent: 2, lockedRace: 5, wallThreat: 2, netWallThreat: 2, wallChain: 1, routeDiversity: 2,
};

interface ArchiveStart { history: GameState[] }

function collectArchive(sourceSeed: number): ArchiveStart[] {
  const result: ArchiveStart[] = [];
  let examined = 0;
  for (let game = 0; examined < archivePositions && result.length < archiveLimit; game++) {
    const rng = rngFrom(sourceSeed + game);
    const history: GameState[] = [initialState()];
    let state = history[0];
    while (!isOver(state) && examined < archivePositions && result.length < archiveLimit) {
      if (state.ply % 4 === game % 4) {
        examined++;
        const move3 = search(state, { maxDepth: 3, weights: champion.weights, repeatPenalty: 0 }).move;
        const move4 = search(state, { maxDepth: 4, weights: champion.weights, repeatPenalty: 0 }).move;
        if (moveToString(move3) !== moveToString(move4)) {
          result.push({ history: history.slice(-10).map(cloneState) });
        }
      }
      const move = search(state, {
        maxDepth: depthAt(state.ply, 3, OPENING_PLIES), weights: champion.weights, margin: 1,
        pickFromTop: pickAt(state.ply, OPENING_PLIES), rng, recent: recentCells(history, state.turn),
      }).move;
      state = applyMove(state, move);
      history.push(state);
    }
  }
  return result;
}

const random = (sourceSeed: number) => rngFrom(sourceSeed);
const pairOpts = (pairSeed: number, start?: ArchiveStart): Omit<MatchConfig, 'weights'> => ({
  depth: 3,
  margin: 1,
  openingPlies: OPENING_PLIES,
  seed: pairSeed,
  startHistory: start?.history,
});

async function train(pool: PairPool, rate: number, archive: ArchiveStart[]): Promise<Weights> {
  let theta = { ...champion.weights };
  for (let iteration = 0; iteration < iterations; iteration++) {
    const deltaRng = random(seed + iteration);
    const delta = Object.fromEntries(TUNED.map((feature) => [feature, deltaRng() < 0.5 ? -1 : 1])) as Record<Feature, number>;
    const plus = { ...theta };
    const minus = { ...theta };
    for (const feature of TUNED) {
      const amount = PERTURB[feature] * 0.5;
      plus[feature] += amount * delta[feature];
      minus[feature] -= amount * delta[feature];
    }
    const matches = Array.from({ length: pairsPerIteration }, (_, index) => {
      // 各率で同じ乱数列を使い、開始局面率以外の条件をそろえる。
      const matchRng = random(seed + 1_000_000 + iteration * 10_000 + index);
      const useArchive = matchRng() < rate;
      const opponent = matchRng() < 0.5 ? champion.weights : league[Math.floor(matchRng() * league.length)];
      const start = archive[Math.floor(matchRng() * archive.length)];
      return {
        opponent,
        opts: pairOpts(seed + iteration * 10_000 + index, useArchive ? start : undefined),
      };
    });
    const [plusScores, minusScores] = await Promise.all([plus, minus].map((weights) =>
      pool.runAll(matches.map((match) => ({ a: weights, b: match.opponent, opts: match.opts }))),
    ));
    const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
    const reward = (sum(plusScores) - sum(minusScores)) / (pairsPerIteration * 2);
    for (const feature of TUNED) theta[feature] += 0.25 * PERTURB[feature] * 0.5 * reward * delta[feature];
  }
  return theta;
}

async function compare(
  pool: PairPool,
  a: Weights,
  b: Weights,
  starts: ArchiveStart[] | null,
  seedOffset: number,
): Promise<number> {
  const pairCount = evalGames / 2;
  const jobs: PairJob[] = Array.from({ length: pairCount }, (_, index) => ({
    a,
    b,
    opts: pairOpts(seed + seedOffset + index, starts ? starts[index % starts.length] : undefined),
  }));
  const scores = await pool.runAll(jobs);
  return scores.reduce((a, b) => a + b, 0) / (pairCount * 2);
}

const percent = (value: number) => `${(100 * value).toFixed(1)}%`;
const trainArchive = collectArchive(97_000_000);
const testArchive = collectArchive(98_000_000);
if (trainArchive.length === 0 || testArchive.length === 0) throw new Error('archive collection produced no positions');

const pool = new PairPool();
try {
  const candidates = new Map<number, Weights>();
  for (const rate of rates) candidates.set(rate, await train(pool, rate, trainArchive));

  console.log(`# 第 ${champion.generation} 世代のアーカイブ開始学習`);
  console.log();
  console.log(`短期学習: 各${iterations}反復 × ${pairsPerIteration}組、開始局面率 ${rates.map((rate) => `${rate * 100}%`).join(' / ')}`);
  console.log(`アーカイブ: 学習${trainArchive.length}局面（seed 97000000）/ 独立評価${testArchive.length}局面（seed 98000000）`);
  console.log(`評価: 各${evalGames}局、3手読み`);
  console.log('\n## チャンピオンに対する勝率');
  console.log('| 開始局面率 | 通常初期局面 | 独立アーカイブ局面 | 初期重み | 重み変化量 |');
  console.log('| ---: | ---: | ---: | ---: | ---: |');
  for (const rate of rates) {
    const weights = candidates.get(rate)!;
    const normal = await compare(pool, weights, champion.weights, null, 100_000 + rate * 10_000);
    const focused = await compare(pool, weights, champion.weights, testArchive, 200_000 + rate * 10_000);
    const initial = await compare(pool, weights, DEFAULT_WEIGHTS, null, 300_000 + rate * 10_000);
    const movement = Math.sqrt(TUNED.reduce((sum, feature) => sum + (weights[feature] - champion.weights[feature]) ** 2, 0));
    console.log(`| ${rate * 100}% | ${percent(normal)} | ${percent(focused)} | ${percent(initial)} | ${movement.toFixed(3)} |`);
  }

  const baseline = candidates.get(0);
  if (baseline) {
    console.log('\n## 通常学習0%との直接比較');
    console.log('| 開始局面率 | 通常初期局面 | 独立アーカイブ局面 |');
    console.log('| ---: | ---: | ---: |');
    for (const rate of rates.filter((value) => value !== 0)) {
      const weights = candidates.get(rate)!;
      const normal = await compare(pool, weights, baseline, null, 400_000 + rate * 10_000);
      const focused = await compare(pool, weights, baseline, testArchive, 500_000 + rate * 10_000);
      console.log(`| ${rate * 100}% | ${percent(normal)} | ${percent(focused)} |`);
    }
  }
  console.log('\n## 判定');
  console.log('- これは小標本の予備比較。通常初期局面と独立アーカイブ局面の両方で0%を上回る率だけを、400局以上の再確認へ進める。');
  console.log('- 改善が片方だけ、または重み変化がほぼ無い場合はTrainへ入れない。');
} finally {
  await pool.close();
}
