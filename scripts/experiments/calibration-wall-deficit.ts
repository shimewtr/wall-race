// 「優勢評価だが相手より壁が少ない」局面を校正に加える価値を、独立標本で確認する。
// 重い自己対戦を含むため、GitHub Actions の Experiment で実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { WIN, withDefaults } from '../../src/engine/ai';
import { calibrationUncertainty } from '../../src/train/calibrate';
import type { GameRecord } from '../../src/train/record';
import { RecordPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '800' },
    depth: { type: 'string', default: '3' },
    seed: { type: 'string', default: '99600000' },
  },
});

const champion = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'));
const games = Number(args.games);
const depth = Number(args.depth);
const seed = Number(args.seed);
const pool = new RecordPool();
const records = await pool.runAll(Array.from({ length: games }, (_, i) => ({
  weights: withDefaults(champion.weights), depth, seed: seed + i,
})));
await pool.close();

interface Sample {
  score: number;
  uncertainty: number;
  wallDiff: number;
  won: number;
}
const samples = (source: GameRecord[]): Sample[] => source.flatMap((game) => game.winner === null ? [] : game.moves
  .filter((move) => Math.abs(move.score) <= WIN / 2)
  .map((move) => ({
    score: move.score,
    uncertainty: calibrationUncertainty(move),
    wallDiff: (move.myWalls - move.opWalls) / 10,
    won: Number(game.winner === move.player),
  })));

const solve = (matrix: number[][], vector: number[]) => {
  const augmented = matrix.map((row, i) => [...row, vector[i]]);
  for (let column = 0; column < vector.length; column++) {
    let pivot = column;
    for (let row = column + 1; row < vector.length; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    if (Math.abs(divisor) < 1e-12) return null;
    for (let j = column; j <= vector.length; j++) augmented[column][j] /= divisor;
    for (let row = 0; row < vector.length; row++) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let j = column; j <= vector.length; j++) augmented[row][j] -= factor * augmented[column][j];
    }
  }
  return augmented.map((row) => row.at(-1)!);
};

interface Model {
  scale: number;
  coefficients: number[];
  deficit: boolean;
}
const features = (sample: Sample, scale: number, deficit: boolean) => {
  const x = sample.score / scale;
  const base = [x, 1, sample.uncertainty, x * sample.uncertainty];
  if (!deficit) return base;
  const favoredDeficit = Math.max(0, -Math.sign(x) * sample.wallDiff);
  return [...base, Math.sign(x) * favoredDeficit, x * favoredDeficit];
};
const fit = (source: Sample[], deficit: boolean): Model => {
  const scale = Math.max(1, Math.sqrt(source.reduce((sum, sample) => sum + sample.score ** 2, 0) / source.length));
  const coefficients = Array.from({ length: deficit ? 6 : 4 }, () => 0);
  for (let iteration = 0; iteration < 50; iteration++) {
    const gradient = coefficients.map(() => 0);
    const hessian = coefficients.map(() => coefficients.map(() => 0));
    for (const sample of source) {
      const fs = features(sample, scale, deficit);
      const z = fs.reduce((sum, value, i) => sum + value * coefficients[i], 0);
      const predicted = 1 / (1 + Math.exp(-z));
      const weight = predicted * (1 - predicted);
      for (let i = 0; i < coefficients.length; i++) {
        gradient[i] += (sample.won - predicted) * fs[i];
        for (let j = 0; j < coefficients.length; j++) hessian[i][j] += weight * fs[i] * fs[j];
      }
    }
    for (let i = 0; i < coefficients.length; i++) hessian[i][i] += 1e-6;
    const delta = solve(hessian, gradient);
    if (!delta) break;
    for (let i = 0; i < coefficients.length; i++) coefficients[i] += delta[i];
    if (delta.reduce((sum, value) => sum + Math.abs(value), 0) < 1e-9) break;
  }
  return { scale, coefficients, deficit };
};
const predict = (sample: Sample, model: Model) => {
  const fs = features(sample, model.scale, model.deficit);
  const z = fs.reduce((sum, value, i) => sum + value * model.coefficients[i], 0);
  return 1 / (1 + Math.exp(-z));
};
const metrics = (source: Sample[], model: Model) => ({
  n: source.length,
  brier: source.reduce((sum, sample) => sum + (predict(sample, model) - sample.won) ** 2, 0) / source.length,
  logLoss: -source.reduce((sum, sample) => {
    const p = Math.max(1e-9, Math.min(1 - 1e-9, predict(sample, model)));
    return sum + sample.won * Math.log(p) + (1 - sample.won) * Math.log(1 - p);
  }, 0) / source.length,
});

const half = Math.floor(records.length / 2);
const train = samples(records.slice(0, half));
const test = samples(records.slice(half));
const current = fit(train, false);
const candidate = fit(train, true);
const favoredDeficit = test.filter((sample) => Math.sign(sample.score) * sample.wallDiff < 0);
const largeDeficit = test.filter((sample) => Math.sign(sample.score) * sample.wallDiff <= -0.2);
const result = {
  generation: champion.generation,
  source: { games, depth, seed, train: train.length, test: test.length },
  current: { model: current, all: metrics(test, current), favoredDeficit: metrics(favoredDeficit, current), largeDeficit: metrics(largeDeficit, current) },
  candidate: { model: candidate, all: metrics(test, candidate), favoredDeficit: metrics(favoredDeficit, candidate), largeDeficit: metrics(largeDeficit, candidate) },
};

console.log(`# 第${champion.generation}世代の壁不足校正`);
console.log(`学習${train.length}局面 / 独立評価${test.length}局面`);
for (const [name, currentMetrics, candidateMetrics] of [
  ['全局面', result.current.all, result.candidate.all],
  ['優勢側の壁不足', result.current.favoredDeficit, result.candidate.favoredDeficit],
  ['優勢側が2枚以上不足', result.current.largeDeficit, result.candidate.largeDeficit],
] as const) {
  console.log(`${name} ${currentMetrics.n}局面: Brier ${currentMetrics.brier.toFixed(4)} → ${candidateMetrics.brier.toFixed(4)}、log loss ${currentMetrics.logLoss.toFixed(4)} → ${candidateMetrics.logLoss.toFixed(4)}`);
}
console.log(`追加係数 [符号付き不足, score×不足]: ${candidate.coefficients.slice(4).map((value) => value.toFixed(4)).join(', ')}`);
writeFileSync('experiment-output.json', `${JSON.stringify(result, null, 2)}\n`);
