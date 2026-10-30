// 深さ別モデルと基準チャンピオンを、同じ歴代世代・seedに当てて一般化を比較する。
// 重い処理なのでGitHub ActionsのExperimentで実行する。

import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { type Weights, withDefaults } from '../../src/engine/ai';
import { OPENING_PLIES } from '../../src/train/match';
import { PairPool, type PairJob } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    model: { type: 'string', default: 'depth4' },
    depth: { type: 'string', default: '4' },
    pairs: { type: 'string', default: '20' },
    seed: { type: 'string', default: '146000000' },
  },
});
const depth = Number(args.depth);
const pairs = Number(args.pairs);
const seed = Number(args.seed);
const root = join(import.meta.dirname, '..', '..');
const trainingDir = join(root, 'training');
const champion = withDefaults(JSON.parse(readFileSync(join(trainingDir, 'champion.json'), 'utf8')).weights);
const modelFile = join(trainingDir, `${args.model}.json`);
const model = withDefaults(JSON.parse(readFileSync(modelFile, 'utf8')).weights);
const generations = readdirSync(join(trainingDir, 'champions'))
  .filter((name) => /^gen-\d+\.json$/.test(name))
  .sort()
  .map((name) => {
    const entry = JSON.parse(readFileSync(join(trainingDir, 'champions', name), 'utf8')) as {
      generation: number;
      weights: Partial<Weights>;
    };
    return { generation: entry.generation, weights: withDefaults(entry.weights) };
  });

interface Matchup {
  generation: number;
  kind: 'model' | 'baseline';
  jobs: PairJob[];
}
const matchups: Matchup[] = generations.flatMap((opponent, generationIndex) =>
  (['model', 'baseline'] as const).map((kind) => ({
    generation: opponent.generation,
    kind,
    jobs: Array.from({ length: pairs }, (_, pairIndex) => ({
      a: kind === 'model' ? model : champion,
      b: opponent.weights,
      opts: {
        depth,
        margin: 1,
        openingPlies: OPENING_PLIES,
        // modelとbaselineは同じ相手・seedで比較する。
        seed: seed + generationIndex * pairs + pairIndex,
      },
    })),
  })),
);
const pool = new PairPool();
const scores = await pool.runAll(matchups.flatMap((matchup) => matchup.jobs));
await pool.close();

let offset = 0;
const rows = matchups.map((matchup) => {
  const values = scores.slice(offset, offset + matchup.jobs.length);
  offset += matchup.jobs.length;
  return {
    generation: matchup.generation,
    kind: matchup.kind,
    games: pairs * 2,
    rate: values.reduce((sum, value) => sum + value, 0) / (pairs * 2),
  };
});
const rate = (generation: number, kind: Matchup['kind']) =>
  rows.find((row) => row.generation === generation && row.kind === kind)!.rate;
const average = (kind: Matchup['kind']) =>
  rows.filter((row) => row.kind === kind).reduce((sum, row) => sum + row.rate, 0) / generations.length;
const modelAverage = average('model');
const baselineAverage = average('baseline');
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

console.log(`# ${args.model}の歴代世代リーグ比較`);
console.log(`深さ${depth}、各世代${pairs * 2}局、合計${generations.length * pairs * 2}局/方式、seed=${seed}`);
console.log(`- ${args.model}: 平均${pct(modelAverage)}`);
console.log(`- champion: 平均${pct(baselineAverage)}`);
console.log(`- 差: ${(modelAverage - baselineAverage) >= 0 ? '+' : ''}${pct(modelAverage - baselineAverage)}`);
console.log('\n| 相手 | model | champion | 差 |');
console.log('| --- | ---: | ---: | ---: |');
for (const opponent of generations) {
  const candidate = rate(opponent.generation, 'model');
  const base = rate(opponent.generation, 'baseline');
  console.log(`| G${opponent.generation} | ${pct(candidate)} | ${pct(base)} | ${candidate - base >= 0 ? '+' : ''}${pct(candidate - base)} |`);
}

writeFileSync('experiment-output.json', `${JSON.stringify({
  source: { model: args.model, depth, pairs, seed },
  modelAverage,
  baselineAverage,
  difference: modelAverage - baselineAverage,
  rows,
}, null, 2)}\n`);
