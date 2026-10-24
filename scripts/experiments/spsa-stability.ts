// SPSA の1反復に使う対局数を増やしたとき、勾配推定のばらつきがどれだけ減るか測る。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/spsa-stability.ts --directions 12 --reference-pairs 128

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { FEATURE_NAMES, type Feature, type Weights, withDefaults } from '../../src/engine/ai';
import { OPENING_PLIES, rngFrom } from '../../src/train/match';
import { PairPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    directions: { type: 'string', default: '12' },
    'reference-pairs': { type: 'string', default: '128' },
  },
});

const directions = Number(args.directions);
const referencePairs = Number(args['reference-pairs']);
const sampleSizes = [16, 32, 64].filter((n) => n <= referencePairs && referencePairs % n === 0);
if (!Number.isInteger(directions) || directions < 2) throw new Error('directions は2以上の整数にしてください');
if (!Number.isInteger(referencePairs) || referencePairs < 16 || referencePairs % 16 !== 0) {
  throw new Error('reference-pairs は16以上かつ16の倍数にしてください');
}

const DIR = join(import.meta.dirname, '..', '..', 'training');
const champion = withDefaults(JSON.parse(readFileSync(join(DIR, 'champion.json'), 'utf8')).weights);
const league: Weights[] = [
  withDefaults(),
  ...readdirSync(join(DIR, 'champions'))
    .filter((f) => f.endsWith('.json'))
    .sort()
    .map((f) => withDefaults(JSON.parse(readFileSync(join(DIR, 'champions', f), 'utf8')).weights)),
];

const fixed = new Set<Feature>(['pathDiff']);
const tuned = FEATURE_NAMES.filter((f) => !fixed.has(f));
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
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const sign = (x: number) => (x > 0 ? 1 : x < 0 ? -1 : 0);

interface Measurement {
  squaredError: number;
  signMatch: boolean;
  referenceNonZero: boolean;
}
const measurements = new Map(sampleSizes.map((n) => [n, [] as Measurement[]]));
const referenceValues: number[] = [];
const pool = new PairPool();
const started = Date.now();

console.log(`第${JSON.parse(readFileSync(join(DIR, 'champion.json'), 'utf8')).generation}世代 / ${directions}方向 / 基準${referencePairs}組`);
console.log(`比較する1反復の対局数: ${sampleSizes.join(' / ')} 組\n`);

try {
  for (let direction = 0; direction < directions; direction++) {
    const rng = rngFrom(80_000_000 + direction);
    const delta = Object.fromEntries(tuned.map((k) => [k, rng() < 0.5 ? -1 : 1])) as Record<Feature, number>;
    const plus = { ...champion };
    const minus = { ...champion };
    for (const k of tuned) {
      const d = perturb[k] * perturbMultiplier * delta[k];
      plus[k] += d;
      minus[k] -= d;
    }

    const matches = Array.from({ length: referencePairs }, (_, i) => ({
      opponent: rng() < 0.5 ? champion : league[Math.floor(rng() * league.length)],
      opts: { depth: 3, margin: 1, openingPlies: OPENING_PLIES, seed: 81_000_000 + direction * referencePairs + i },
    }));
    const [plusScores, minusScores] = await Promise.all(
      [plus, minus].map((weights) =>
        pool.runAll(matches.map((m) => ({ a: weights, b: m.opponent, opts: m.opts }))),
      ),
    );
    const differences = plusScores.map((score, i) => score - minusScores[i]);
    const reference = sum(differences) / (referencePairs * 2);
    referenceValues.push(reference);

    for (const size of sampleSizes) {
      for (let offset = 0; offset < referencePairs; offset += size) {
        const estimate = sum(differences.slice(offset, offset + size)) / (size * 2);
        measurements.get(size)!.push({
          squaredError: (estimate - reference) ** 2,
          signMatch: sign(estimate) === sign(reference),
          referenceNonZero: sign(reference) !== 0,
        });
      }
    }
    console.log(`方向 ${String(direction + 1).padStart(2)}: 基準勾配 ${reference.toFixed(4)}`);
  }
} finally {
  await pool.close();
}

console.log('\n対局数  RMSE     基準と同符号  相対所要時間');
for (const size of sampleSizes) {
  const rows = measurements.get(size)!;
  const nonZero = rows.filter((r) => r.referenceNonZero);
  const rmse = Math.sqrt(sum(rows.map((r) => r.squaredError)) / rows.length);
  const agreement = nonZero.filter((r) => r.signMatch).length / nonZero.length;
  console.log(`${String(size).padStart(6)}  ${rmse.toFixed(4)}   ${(agreement * 100).toFixed(1).padStart(5)}%       ${(size / 16).toFixed(1)}x`);
}
const meanAbsReference = sum(referenceValues.map(Math.abs)) / referenceValues.length;
console.log(`\n基準勾配の平均絶対値: ${meanAbsReference.toFixed(4)}`);
console.log(`所要時間: ${((Date.now() - started) / 60_000).toFixed(1)}分`);
