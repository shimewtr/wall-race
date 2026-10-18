// 手引き AI の強さを測る。手引きの書き方（パラメータ）の違いごとに、強さの違う相手との勝率を出す。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/guide.ts --games 100

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { withDefaults } from '../../src/engine/ai';
import type { GuideJob } from '../../src/guide/match';
import { GUIDE_CURRENT, GUIDE_V1, GUIDE_V2, GUIDE_V3, type GuideParams } from '../../src/guide/policy';
import { GuidePool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '100' },
    // 追加で試すパラメータ（JSON。現行の手引きとの差分）
    variant: { type: 'string', multiple: true, default: [] },
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
  ['第 2 版 + 壁 1 枚 = 2 歩で形勢を数える', { ...GUIDE_V2, wallValue: 2 }],
  ...args.variant!.map((v): [string, GuideParams] => [v, { ...GUIDE_CURRENT, ...JSON.parse(v) }]),
];

const opponents = [
  { name: '初期の重み・1 手読み', weights: withDefaults(), depth: 1 },
  { name: 'チャンピオン・2 手読み', weights: championWeights, depth: 2 },
  { name: 'チャンピオン・3 手読み', weights: championWeights, depth: 3 },
];

const games = Number(args.games);
const pool = new GuidePool();
console.log(`手引き AI の勝率（相手ごとに ${games} 局、先後半々）\n`);
console.log(`| 手引き | ${opponents.map((o) => o.name).join(' | ')} |`);
console.log(`| --- | ${opponents.map(() => '---').join(' | ')} |`);
for (const [name, params] of variants) {
  const cells: string[] = [];
  for (const o of opponents) {
    const jobs: GuideJob[] = Array.from({ length: games }, (_, i) => ({
      params,
      weights: o.weights,
      depth: o.depth,
      guidePlayer: (i % 2) as 0 | 1,
      seed: 13_000_000 + i,
    }));
    const scores = await pool.runAll(jobs);
    cells.push(`${((scores.reduce((a, b) => a + b, 0) / games) * 100).toFixed(0)}%`);
  }
  console.log(`| ${name} | ${cells.join(' | ')} |`);
}
await pool.close();
