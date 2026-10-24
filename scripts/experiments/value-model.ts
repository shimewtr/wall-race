// 現在の静的評価特徴量から勝敗を予測し、終盤度・残り壁との非線形な関係に価値があるか調べる。
// 対局生成と全特徴量の再計算が重いため、GitHub Actions の Experiment で実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { FEATURE_NAMES, WIN, features, withDefaults } from '../../src/engine/ai';
import { applyMove, initialState, moveFromString, shortestDistance } from '../../src/engine/game';
import { calibrationUncertainty, type Calibration, winRate } from '../../src/train/calibrate';
import type { GameRecord, MoveRecord } from '../../src/train/record';
import { RecordPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '800' },
    depth: { type: 'string', default: '3' },
    seed: { type: 'string', default: '99700000' },
    every: { type: 'string', default: '4' },
  },
});

const ROOT = join(import.meta.dirname, '..', '..');
const champion = JSON.parse(readFileSync(join(ROOT, 'training', 'champion.json'), 'utf8'));
const calibration = JSON.parse(readFileSync(join(ROOT, 'training', 'calibration.json'), 'utf8')) as Calibration;
const games = Number(args.games);
const depth = Number(args.depth);
const seed = Number(args.seed);
const every = Number(args.every);

const pool = new RecordPool();
const records = await pool.runAll(Array.from({ length: games }, (_, i) => ({
  weights: withDefaults(champion.weights), depth, seed: seed + i,
})));
await pool.close();

interface Sample {
  raw: number[];
  interactions: number[];
  current: number;
  won: number;
  phase: 'opening' | 'middle' | 'endgame';
  walls: 'low' | 'middle' | 'high';
}

const interactionNames = [
  'pathDiff×totalWalls',
  'wallDiff×totalWalls',
  'netWallThreat×totalWalls',
  'pathDiff×closeness',
  'wallDiff×closeness',
  'wallThreat×closeness',
  'netWallThreat×closeness',
  'wallChain×closeness',
  'routeDiversity×closeness',
  'pathDiff×abs(wallDiff)',
  'wallDiff×abs(pathDiff)',
  'netWallThreat×abs(pathDiff)',
] as const;

function samplesFrom(source: GameRecord[]): Sample[] {
  const result: Sample[] = [];
  for (const game of source) {
    if (game.winner === null) continue;
    let state = initialState();
    for (let i = 0; i < game.moves.length; i++) {
      const recorded = game.moves[i];
      if (i % every === 0 && Math.abs(recorded.score) <= WIN / 2) {
        const myDist = shortestDistance(state, state.turn);
        const opDist = shortestDistance(state, (1 - state.turn) as 0 | 1);
        const f = features(state, myDist, opDist);
        const raw = FEATURE_NAMES.map((name) => f[name]);
        const totalWalls = (state.wallsLeft[0] + state.wallsLeft[1]) / 20;
        const closeness = 1 / (Math.min(myDist, opDist) + 1);
        result.push({
          raw,
          interactions: [
            f.pathDiff * totalWalls,
            f.wallDiff * totalWalls,
            f.netWallThreat * totalWalls,
            f.pathDiff * closeness,
            f.wallDiff * closeness,
            f.wallThreat * closeness,
            f.netWallThreat * closeness,
            f.wallChain * closeness,
            f.routeDiversity * closeness,
            f.pathDiff * Math.abs(f.wallDiff),
            f.wallDiff * Math.abs(f.pathDiff),
            f.netWallThreat * Math.abs(f.pathDiff),
          ],
          current: winRate(recorded.score, calibration, calibrationUncertainty(recorded)),
          won: Number(game.winner === state.turn),
          phase: state.ply < 12 ? 'opening' : Math.min(myDist, opDist) <= 4 ? 'endgame' : 'middle',
          walls: totalWalls <= 0.3 ? 'low' : totalWalls <= 0.65 ? 'middle' : 'high',
        });
      }
      const move = moveFromString(recorded.move);
      if (!move) throw new Error(`invalid move: ${recorded.move}`);
      state = applyMove(state, move);
    }
  }
  return result;
}

const solve = (matrix: number[][], vector: number[]) => {
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let column = 0; column < vector.length; column++) {
    let pivot = column;
    for (let row = column + 1; row < vector.length; row++) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    [a[column], a[pivot]] = [a[pivot], a[column]];
    const divisor = a[column][column];
    if (Math.abs(divisor) < 1e-10) return null;
    for (let j = column; j <= vector.length; j++) a[column][j] /= divisor;
    for (let row = 0; row < vector.length; row++) {
      if (row === column) continue;
      const factor = a[row][column];
      for (let j = column; j <= vector.length; j++) a[row][j] -= factor * a[column][j];
    }
  }
  return a.map((row) => row.at(-1)!);
};

interface Model {
  names: string[];
  means: number[];
  scales: number[];
  coefficients: number[];
}

const vector = (sample: Sample, nonlinear: boolean) => nonlinear ? [...sample.raw, ...sample.interactions] : sample.raw;
function fit(source: Sample[], nonlinear: boolean): Model {
  const names = nonlinear ? [...FEATURE_NAMES, ...interactionNames] : [...FEATURE_NAMES];
  const width = names.length;
  const means = Array.from({ length: width }, (_, i) => source.reduce((sum, s) => sum + vector(s, nonlinear)[i], 0) / source.length);
  const scales = means.map((mean, i) => Math.max(1e-6, Math.sqrt(source.reduce((sum, s) => sum + (vector(s, nonlinear)[i] - mean) ** 2, 0) / source.length)));
  const xs = source.map((sample) => [1, ...vector(sample, nonlinear).map((value, i) => (value - means[i]) / scales[i])]);
  const coefficients = Array.from({ length: width + 1 }, () => 0);
  const lambda = 1;
  for (let iteration = 0; iteration < 50; iteration++) {
    const gradient = coefficients.map(() => 0);
    const hessian = coefficients.map(() => coefficients.map(() => 0));
    for (let row = 0; row < source.length; row++) {
      const x = xs[row];
      const z = x.reduce((sum, value, i) => sum + value * coefficients[i], 0);
      const p = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
      const weight = p * (1 - p);
      for (let i = 0; i < coefficients.length; i++) {
        gradient[i] += (source[row].won - p) * x[i];
        for (let j = 0; j < coefficients.length; j++) hessian[i][j] += weight * x[i] * x[j];
      }
    }
    for (let i = 1; i < coefficients.length; i++) {
      gradient[i] -= lambda * coefficients[i];
      hessian[i][i] += lambda;
    }
    hessian[0][0] += 1e-8;
    const delta = solve(hessian, gradient);
    if (!delta) break;
    for (let i = 0; i < coefficients.length; i++) coefficients[i] += delta[i];
    if (delta.reduce((sum, value) => sum + Math.abs(value), 0) < 1e-8) break;
  }
  return { names, means, scales, coefficients };
}

function predict(sample: Sample, model: Model) {
  const nonlinear = model.names.length > FEATURE_NAMES.length;
  const x = vector(sample, nonlinear);
  const z = model.coefficients[0] + x.reduce((sum, value, i) => sum + ((value - model.means[i]) / model.scales[i]) * model.coefficients[i + 1], 0);
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
}

interface Metrics { n: number; brier: number; logLoss: number; highConfidenceError: number }
function metrics(source: Sample[], getPrediction: (sample: Sample) => number): Metrics {
  if (source.length === 0) return { n: 0, brier: 0, logLoss: 0, highConfidenceError: 0 };
  let brier = 0;
  let logLoss = 0;
  let errors = 0;
  for (const sample of source) {
    const p = Math.max(1e-9, Math.min(1 - 1e-9, getPrediction(sample)));
    brier += (p - sample.won) ** 2;
    logLoss -= sample.won * Math.log(p) + (1 - sample.won) * Math.log(1 - p);
    if ((p >= 0.8 && sample.won === 0) || (p <= 0.2 && sample.won === 1)) errors++;
  }
  return { n: source.length, brier: brier / source.length, logLoss: logLoss / source.length, highConfidenceError: errors / source.length };
}

const half = Math.floor(records.length / 2);
const train = samplesFrom(records.slice(0, half));
const test = samplesFrom(records.slice(half));
const linear = fit(train, false);
const nonlinear = fit(train, true);
const evaluate = (source: Sample[]) => ({
  current: metrics(source, (sample) => sample.current),
  linear: metrics(source, (sample) => predict(sample, linear)),
  nonlinear: metrics(source, (sample) => predict(sample, nonlinear)),
});
const result = {
  generation: champion.generation,
  source: { games, depth, seed, every, train: train.length, test: test.length },
  all: evaluate(test),
  phase: Object.fromEntries(['opening', 'middle', 'endgame'].map((name) => [name, evaluate(test.filter((sample) => sample.phase === name))])),
  walls: Object.fromEntries(['low', 'middle', 'high'].map((name) => [name, evaluate(test.filter((sample) => sample.walls === name))])),
  models: { linear, nonlinear },
};

const printRow = (name: string, value: ReturnType<typeof evaluate>) => {
  console.log(`${name} (${value.current.n}局面):`);
  console.log(`  現行校正   Brier ${value.current.brier.toFixed(4)} / log loss ${value.current.logLoss.toFixed(4)} / 高信頼誤り ${(value.current.highConfidenceError * 100).toFixed(1)}%`);
  console.log(`  線形特徴量 Brier ${value.linear.brier.toFixed(4)} / log loss ${value.linear.logLoss.toFixed(4)} / 高信頼誤り ${(value.linear.highConfidenceError * 100).toFixed(1)}%`);
  console.log(`  交互作用   Brier ${value.nonlinear.brier.toFixed(4)} / log loss ${value.nonlinear.logLoss.toFixed(4)} / 高信頼誤り ${(value.nonlinear.highConfidenceError * 100).toFixed(1)}%`);
};
console.log(`# 第${champion.generation}世代の価値モデル実験`);
console.log(`ゲーム単位で学習${half}局/評価${games - half}局、${every}手ごと: ${train.length}/${test.length}局面`);
printRow('全局面', result.all);
for (const [name, value] of Object.entries(result.phase)) printRow(`phase=${name}`, value);
for (const [name, value] of Object.entries(result.walls)) printRow(`walls=${name}`, value);
const interactionCoefficients = interactionNames.map((name, i) => ({
  name,
  coefficient: nonlinear.coefficients[FEATURE_NAMES.length + i + 1],
})).sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient));
console.log('交互作用の標準化係数:');
for (const item of interactionCoefficients) console.log(`  ${item.name}: ${item.coefficient.toFixed(4)}`);
writeFileSync('experiment-output.json', `${JSON.stringify({ ...result, interactionCoefficients }, null, 2)}\n`);
