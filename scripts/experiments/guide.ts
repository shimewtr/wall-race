// 手引き AI の強さを測る。手引きの書き方（パラメータ）の違いごとに、強さの違う相手との勝率を出す。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/guide.ts --games 100

import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { withDefaults } from '../../src/engine/ai';
import type { GuideJob } from '../../src/guide/match';
import { GUIDE_CURRENT, GUIDE_V1, GUIDE_V2, GUIDE_V3, GUIDE_V4, type GuideParams } from '../../src/guide/policy';
import { GuidePool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '100' },
    // 追加で試すパラメータ（JSON。現行の手引きとの差分）
    variant: { type: 'string', multiple: true, default: [] },
    // league: 過去の全世代 × 2・3 手読みを相手にして平均を出す（特定の相手への合わせ込みを避ける）
    opponents: { type: 'string', default: 'current' },
  },
});

const CHAMPION = join(import.meta.dirname, '..', '..', 'training', 'champion.json');
const championWeights = withDefaults(
  existsSync(CHAMPION) ? JSON.parse(readFileSync(CHAMPION, 'utf8')).weights : {},
);

const variants: Array<[string, GuideParams]> = [
  ['手引き初版', GUIDE_V1],
  ['第 2 版（相手の最善の邪魔を 1 手想定）', GUIDE_V2],
  ['第 3 版（相手も第 2 版で考えると想定）', GUIDE_V3],
  ['第 4 版（第 2 版 + 返しのあとの遠回りの差の半分）', GUIDE_V4],
  ...args.variant!.map((v): [string, GuideParams] => [v, { ...GUIDE_CURRENT, ...JSON.parse(v) }]),
];

const CHAMPIONS = join(import.meta.dirname, '..', '..', 'training', 'champions');
const league = () =>
  readdirSync(CHAMPIONS)
    .filter((f) => f.endsWith('.json'))
    .sort()
    .flatMap((f) => {
      const c = JSON.parse(readFileSync(join(CHAMPIONS, f), 'utf8'));
      return [2, 3].map((depth) => ({ name: `第${c.generation}世代・${depth}手`, weights: withDefaults(c.weights), depth }));
    });

const opponents =
  args.opponents === 'league'
    ? league()
    : [
        { name: '初期の重み・1 手読み', weights: withDefaults(), depth: 1 },
        { name: 'チャンピオン・2 手読み', weights: championWeights, depth: 2 },
        { name: 'チャンピオン・3 手読み', weights: championWeights, depth: 3 },
      ];

const games = Number(args.games);
const pool = new GuidePool();
console.log(`手引き AI の勝率（相手ごとに ${games} 局、先後半々）\n`);
console.log(`| 手引き | ${opponents.map((o) => o.name).join(' | ')} | 平均 |`);
console.log(`| --- | ${opponents.map(() => '---').join(' | ')} | --- |`);
for (const [name, params] of variants) {
  const cells: string[] = [];
  let sum = 0;
  for (const o of opponents) {
    const jobs: GuideJob[] = Array.from({ length: games }, (_, i) => ({
      params,
      weights: o.weights,
      depth: o.depth,
      guidePlayer: (i % 2) as 0 | 1,
      seed: 13_000_000 + i,
    }));
    const scores = await pool.runAll(jobs);
    const rate = scores.reduce((a, b) => a + b, 0) / games;
    sum += rate;
    cells.push(`${(rate * 100).toFixed(0)}%`);
  }
  console.log(`| ${name} | ${cells.join(' | ')} | **${((sum / opponents.length) * 100).toFixed(1)}%** |`);
}
await pool.close();
