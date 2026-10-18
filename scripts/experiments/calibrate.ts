// チャンピオンの評価値を勝率に換算する式を求め、当てはまりを確かめる。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
// 出力の JSON を training/calibration.json に保存すると Web の形勢表示が勝率になる
// （以降は学習で昇格したときに自動で作り直される）。
//
// 例: npx tsx scripts/experiments/calibrate.ts --games 300

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { withDefaults } from '../../src/engine/ai';
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

// 半分で式を求め、残り半分で当てはまりを確かめる
const records = await pool.runAll(Array.from({ length: games }, (_, i) => job(i)));
await pool.close();
const half = Math.floor(games / 2);
const fit = fitCalibration(records.slice(0, half), champion.generation);
console.log(`第 ${champion.generation} 世代: a=${fit.a.toPrecision(4)} b=${fit.b.toFixed(3)}（${fit.samples} 局面で当てはめ）`);

const buckets = Array.from({ length: 10 }, () => ({ n: 0, won: 0 }));
for (const g of records.slice(half)) {
  if (g.winner === null) continue;
  for (const m of g.moves) {
    const p = winRate(m.score, fit);
    const b = buckets[Math.min(9, Math.floor(p * 10))];
    b.n++;
    if (g.winner === m.player) b.won++;
  }
}
console.log('\n残り半分の対局での確認（換算した勝率 → 実際に勝った割合）:');
buckets.forEach((b, i) => {
  if (b.n > 0) console.log(`  ${i * 10}〜${i * 10 + 10}%: ${((b.won / b.n) * 100).toFixed(1)}%（${b.n} 局面）`);
});

const all = fitCalibration(records, champion.generation);
console.log('\ntraining/calibration.json に保存する内容（全対局で当てはめ）:');
console.log(JSON.stringify(all));
