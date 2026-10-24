// チャンピオンの評価値を勝率に換算する式を求め、当てはまりを確かめる。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
// 出力の JSON を training/calibration.json に保存すると Web の形勢表示が勝率になる
// （以降は学習で昇格したときに自動で作り直される）。
//
// 例: npx tsx scripts/experiments/calibrate.ts --games 300

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { WIN, withDefaults } from '../../src/engine/ai';
import { fitCalibration, winRate } from '../../src/train/calibrate';
import { RecordPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '300' },
    depth: { type: 'string', default: '3' },
    seed: { type: 'string', default: '31000000' },
  },
});

const CHAMPION = join(import.meta.dirname, '..', '..', 'training', 'champion.json');
const champion = existsSync(CHAMPION) ? JSON.parse(readFileSync(CHAMPION, 'utf8')) : { generation: 0, weights: {} };
const pool = new RecordPool();
const games = Number(args.games);
const base = Number(args.seed);
const job = (i: number) => ({ weights: withDefaults(champion.weights), depth: Number(args.depth), seed: base + i });

interface ContextCalibration {
  scale: number;
  // [score, intercept, uncertainty, score × uncertainty]
  coefficients: [number, number, number, number];
}

const uncertainty = (move: { myDist: number; opDist: number; myWalls: number; opWalls: number }) =>
  ((move.myWalls + move.opWalls) / 20) * Math.max(0, Math.min(1, (6 - Math.min(move.myDist, move.opDist)) / 5));

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

const fitContextCalibration = (source: typeof records): ContextCalibration => {
  const samples = source.flatMap((game) => game.winner === null ? [] : game.moves
    .filter((move) => Math.abs(move.score) <= WIN / 2)
    .map((move) => ({ score: move.score, u: uncertainty(move), won: Number(game.winner === move.player) })));
  const scale = Math.max(1, Math.sqrt(samples.reduce((sum, sample) => sum + sample.score ** 2, 0) / samples.length));
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

const contextWinRate = (score: number, u: number, calibration: ContextCalibration) => {
  if (score > WIN / 2) return 1;
  if (score < -WIN / 2) return 0;
  const x = score / calibration.scale;
  const features = [x, 1, u, x * u];
  const z = features.reduce((sum, feature, i) => sum + feature * calibration.coefficients[i], 0);
  return 1 / (1 + Math.exp(-z));
};

// 半分で式を求め、残り半分で当てはまりを確かめる
const records = await pool.runAll(Array.from({ length: games }, (_, i) => job(i)));
await pool.close();
const half = Math.floor(games / 2);
const fit = fitCalibration(records.slice(0, half), champion.generation);
const contextFit = fitContextCalibration(records.slice(0, half));
console.log(`第 ${champion.generation} 世代: a=${fit.a.toPrecision(4)} b=${fit.b.toFixed(3)}（${fit.samples} 局面で当てはめ）`);

const buckets = Array.from({ length: 10 }, () => ({ n: 0, won: 0 }));
const observations: Array<{
  seed: number;
  ply: number;
  move: string;
  predicted: number;
  contextPredicted: number;
  uncertainty: number;
  won: number;
  phase: '序盤' | '中盤' | '終盤';
  walls: '壁0〜6枚' | '壁7〜13枚' | '壁14〜20枚';
  race: '2歩以上先行' | 'ほぼ互角' | '2歩以上遅れ';
  myDist: number;
  opDist: number;
  myWalls: number;
  opWalls: number;
}> = [];
for (const [gameIndex, g] of records.slice(half).entries()) {
  if (g.winner === null) continue;
  for (const m of g.moves) {
    const p = winRate(m.score, fit);
    const u = uncertainty(m);
    const b = buckets[Math.min(9, Math.floor(p * 10))];
    b.n++;
    const won = Number(g.winner === m.player);
    if (won) b.won++;
    const totalWalls = m.myWalls + m.opWalls;
    observations.push({
      seed: base + half + gameIndex,
      ply: m.ply,
      move: m.move,
      predicted: p,
      contextPredicted: contextWinRate(m.score, u, contextFit),
      uncertainty: u,
      won,
      phase: m.ply < 12 ? '序盤' : Math.min(m.myDist, m.opDist) <= 4 ? '終盤' : '中盤',
      walls: totalWalls <= 6 ? '壁0〜6枚' : totalWalls <= 13 ? '壁7〜13枚' : '壁14〜20枚',
      race: m.opDist - m.myDist >= 2 ? '2歩以上先行' : m.myDist - m.opDist >= 2 ? '2歩以上遅れ' : 'ほぼ互角',
      myDist: m.myDist,
      opDist: m.opDist,
      myWalls: m.myWalls,
      opWalls: m.opWalls,
    });
  }
}
console.log('\n残り半分の対局での確認（換算した勝率 → 実際に勝った割合）:');
buckets.forEach((b, i) => {
  if (b.n > 0) console.log(`  ${i * 10}〜${i * 10 + 10}%: ${((b.won / b.n) * 100).toFixed(1)}%（${b.n} 局面）`);
});

const summarize = (items: typeof observations) => ({
  n: items.length,
  predicted: items.reduce((sum, item) => sum + item.predicted, 0) / items.length,
  actual: items.reduce((sum, item) => sum + item.won, 0) / items.length,
  brier: items.reduce((sum, item) => sum + (item.predicted - item.won) ** 2, 0) / items.length,
});
const scoreModel = (items: typeof observations, key: 'predicted' | 'contextPredicted') => ({
  brier: items.reduce((sum, item) => sum + (item[key] - item.won) ** 2, 0) / items.length,
  logLoss: -items.reduce((sum, item) => {
    const p = Math.max(1e-9, Math.min(1 - 1e-9, item[key]));
    return sum + item.won * Math.log(p) + (1 - item.won) * Math.log(1 - p);
  }, 0) / items.length,
});
const baseMetrics = scoreModel(observations, 'predicted');
const contextMetrics = scoreModel(observations, 'contextPredicted');
const volatile = observations.filter((item) => item.uncertainty >= 0.4);
const volatileBaseMetrics = scoreModel(volatile, 'predicted');
const volatileContextMetrics = scoreModel(volatile, 'contextPredicted');
console.log('\n壁数×ゴールの近さを加えた校正式との独立比較:');
console.log(`  全局面 Brier: ${baseMetrics.brier.toFixed(4)} → ${contextMetrics.brier.toFixed(4)}、log loss: ${baseMetrics.logLoss.toFixed(4)} → ${contextMetrics.logLoss.toFixed(4)}`);
console.log(`  高不確実性${volatile.length}局面 Brier: ${volatileBaseMetrics.brier.toFixed(4)} → ${volatileContextMetrics.brier.toFixed(4)}、log loss: ${volatileBaseMetrics.logLoss.toFixed(4)} → ${volatileContextMetrics.logLoss.toFixed(4)}`);
console.log(`  係数 [score, intercept, uncertainty, score×uncertainty]: ${contextFit.coefficients.map((value) => value.toFixed(4)).join(', ')}（score scale ${contextFit.scale.toFixed(2)}）`);
const grouped = (key: 'phase' | 'walls' | 'race') =>
  [...new Set(observations.map((item) => item[key]))].map((name) => ({
    name,
    ...summarize(observations.filter((item) => item[key] === name)),
  }));
const groups = {
  phase: grouped('phase'),
  walls: grouped('walls'),
  race: grouped('race'),
};
const printGroups = (title: string, items: Array<ReturnType<typeof summarize> & { name: string }>) => {
  console.log(`\n${title}（予測 → 実勝率、差、Brier）:`);
  for (const item of items) {
    const gap = item.actual - item.predicted;
    console.log(`  ${item.name}: ${(item.predicted * 100).toFixed(1)}% → ${(item.actual * 100).toFixed(1)}%、${gap >= 0 ? '+' : ''}${(gap * 100).toFixed(1)}pt、${item.brier.toFixed(3)}（${item.n}局面）`);
  }
};
printGroups('フェーズ別', groups.phase);
printGroups('盤上の残り壁別', groups.walls);
printGroups('最短距離差別', groups.race);

const highConfidenceErrors = observations
  .filter((item) => (item.predicted >= 0.8 && item.won === 0) || (item.predicted <= 0.2 && item.won === 1))
  .sort((a, b) => Math.abs(b.predicted - b.won) - Math.abs(a.predicted - a.won));
console.log(`\n高信頼（80%以上）で外した局面: ${highConfidenceErrors.length}/${observations.length}（${((highConfidenceErrors.length / observations.length) * 100).toFixed(1)}%）`);
for (const item of highConfidenceErrors.slice(0, 12)) {
  console.log(`  seed=${item.seed} ply=${item.ply} ${item.move}: 予測${(item.predicted * 100).toFixed(1)}% → ${item.won ? '勝ち' : '負け'}、${item.phase}、距離${item.myDist}-${item.opDist}、壁${item.myWalls}-${item.opWalls}`);
}

const all = fitCalibration(records, champion.generation);
console.log('\ntraining/calibration.json に保存する内容（全対局で当てはめ）:');
console.log(JSON.stringify(all));
writeFileSync('experiment-output.json', `${JSON.stringify({
  generation: champion.generation,
  games,
  depth: Number(args.depth),
  seed: base,
  validation: summarize(observations),
  modelComparison: { base: baseMetrics, context: contextMetrics, volatile: { n: volatile.length, base: volatileBaseMetrics, context: volatileContextMetrics } },
  contextCalibration: contextFit,
  buckets,
  groups,
  highConfidenceErrors,
  calibration: all,
}, null, 2)}\n`);
