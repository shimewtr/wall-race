// 探索の評価値を「勝率」に換算するための式を、自己対戦の結果から求める。
// 評価値は重みの組み合わせなので「何歩分」とは読めない。そこで
//   P(手番側が勝つ) = 1 / (1 + exp(-(a × 評価値 + b)))
// のロジスティック回帰で、実際の勝ち負けに合う a, b を求める。b は読みの深さの偶奇などによる手番側の片寄りを吸収する。

import { WIN } from '../engine/ai';
import type { GameRecord } from './record';

export interface Calibration {
  generation: number;
  a: number;
  b: number;
  // 当てはめに使った局面数
  samples: number;
  // 壁を多く残してゴールへ近づいた局面では、同じ評価差でも逆転が多い。
  // scoreだけの式より独立標本のBrier scoreが改善した場合だけ付与する。
  context?: {
    scale: number;
    // [score, intercept, uncertainty, score × uncertainty]
    coefficients: [number, number, number, number];
  };
}

export const winRate = (score: number, cal: Pick<Calibration, 'a' | 'b' | 'context'>, uncertainty?: number): number => {
  if (score > WIN / 2) return 1;
  if (score < -WIN / 2) return 0;
  if (cal.context && uncertainty !== undefined) {
    const x = score / cal.context.scale;
    const [scoreWeight, intercept, uncertaintyWeight, interaction] = cal.context.coefficients;
    return 1 / (1 + Math.exp(-(scoreWeight * x + intercept + uncertaintyWeight * uncertainty + interaction * x * uncertainty)));
  }
  return 1 / (1 + Math.exp(-(cal.a * score + cal.b)));
};

export const calibrationUncertainty = (move: Pick<GameRecord['moves'][number], 'myDist' | 'opDist' | 'myWalls' | 'opWalls'>) => {
  if (![move.myDist, move.opDist, move.myWalls, move.opWalls].every(Number.isFinite)) return 0;
  return ((move.myWalls + move.opWalls) / 20) * Math.max(0, Math.min(1, (6 - Math.min(move.myDist, move.opDist)) / 5));
};

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

const fitContext = (records: GameRecord[]): NonNullable<Calibration['context']> => {
  const samples = records.flatMap((game) => game.winner === null ? [] : game.moves
    .filter((move) => Math.abs(move.score) <= WIN / 2)
    .map((move) => ({ score: move.score, u: calibrationUncertainty(move), won: Number(game.winner === move.player) })));
  const scale = Math.max(1, Math.sqrt(samples.reduce((sum, sample) => sum + sample.score ** 2, 0) / Math.max(1, samples.length)));
  const coefficients: [number, number, number, number] = [0, 0, 0, 0];
  for (let iteration = 0; iteration < 50; iteration++) {
    const gradient = [0, 0, 0, 0];
    const hessian = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
    for (const sample of samples) {
      const x = sample.score / scale;
      const features = [x, 1, sample.u, x * sample.u];
      const z = features.reduce((sum, feature, i) => sum + feature * coefficients[i], 0);
      const predicted = 1 / (1 + Math.exp(-z));
      const weight = predicted * (1 - predicted);
      for (let i = 0; i < 4; i++) {
        gradient[i] += (sample.won - predicted) * features[i];
        for (let j = 0; j < 4; j++) hessian[i][j] += weight * features[i] * features[j];
      }
    }
    for (let i = 0; i < 4; i++) hessian[i][i] += 1e-6;
    const delta = solve(hessian, gradient);
    if (!delta) break;
    for (let i = 0; i < 4; i++) coefficients[i] += delta[i];
    if (delta.reduce((sum, value) => sum + Math.abs(value), 0) < 1e-9) break;
  }
  return { scale, coefficients };
};

// 自己対戦の記録から (評価値, 手番側が勝ったか) を集めてロジスティック回帰する（ニュートン法）
export function fitCalibration(records: GameRecord[], generation: number): Calibration {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const g of records) {
    if (g.winner === null) continue;
    for (const m of g.moves) {
      if (Math.abs(m.score) > WIN / 2) continue; // 勝ちを読み切った局面は換算しなくてよい
      xs.push(m.score);
      ys.push(g.winner === m.player ? 1 : 0);
    }
  }
  // 評価値は数百点の幅があるので、尺度をそろえてから解く
  const scale = Math.max(1, Math.sqrt(xs.reduce((s, x) => s + x * x, 0) / Math.max(1, xs.length)));
  let a = 0;
  let b = 0;
  for (let iter = 0; iter < 50; iter++) {
    let ga = 0;
    let gb = 0;
    let haa = 0;
    let hab = 0;
    let hbb = 0;
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i] / scale;
      const p = 1 / (1 + Math.exp(-(a * x + b)));
      const w = p * (1 - p);
      ga += (ys[i] - p) * x;
      gb += ys[i] - p;
      haa += w * x * x;
      hab += w * x;
      hbb += w;
    }
    const det = haa * hbb - hab * hab;
    if (det <= 1e-12) break;
    const da = (hbb * ga - hab * gb) / det;
    const db = (haa * gb - hab * ga) / det;
    a += da;
    b += db;
    if (Math.abs(da) + Math.abs(db) < 1e-9) break;
  }
  return { generation, a: a / scale, b, samples: xs.length, context: fitContext(records) };
}
