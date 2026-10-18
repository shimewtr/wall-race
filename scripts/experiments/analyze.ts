// チャンピオン同士の対局を記録して、手引きの根拠になる統計を出す。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/analyze.ts --games 300

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { withDefaults } from '../../src/engine/ai';
import { OPENING_PLIES } from '../../src/train/match';
import type { GameRecord, MoveRecord } from '../../src/train/record';
import { RecordPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '300' },
    depth: { type: 'string', default: '3' },
  },
});

const CHAMPION = join(import.meta.dirname, '..', '..', 'training', 'champion.json');
const champion = existsSync(CHAMPION) ? JSON.parse(readFileSync(CHAMPION, 'utf8')) : { generation: 0, weights: {} };
const weights = withDefaults(champion.weights);
const games = Number(args.games);

const pool = new RecordPool();
const records: GameRecord[] = await pool.runAll(
  Array.from({ length: games }, (_, i) => ({ weights, depth: Number(args.depth), seed: 11_000_000 + i })),
);
await pool.close();

const pct = (n: number, d: number) => (d === 0 ? '  -  ' : `${((n / d) * 100).toFixed(1).padStart(5)}%`);
const avg = (xs: number[]) => (xs.length === 0 ? NaN : xs.reduce((a, b) => a + b, 0) / xs.length);
const f1 = (x: number) => x.toFixed(1);
const lines: string[] = [];
const out = (s = '') => lines.push(s);
const all = records.flatMap((g) => g.moves.map((m) => ({ ...m, won: g.winner === m.player })));

out(`# 第 ${champion.generation} 世代の自己対戦 ${games} 局の分析（${args.depth} 手読み）`);
out();
const decided = records.filter((g) => g.winner !== null);
out(`先手の勝率 ${pct(decided.filter((g) => g.winner === 0).length, decided.length)} / 平均手数 ${f1(avg(records.map((g) => g.moves.length)))} / 引き分け ${records.length - decided.length} 局`);

// ---- 壁を使うタイミング ----
out('\n## 壁を使うタイミング');
const firstWall = records.flatMap((g) =>
  [0, 1].map((p) => g.moves.find((m) => m.player === p && m.type === 'wall')).filter((m): m is MoveRecord => !!m),
);
out(`最初の壁: 平均 ${f1(avg(firstWall.map((m) => m.ply)))} 手目、そのときの相手の残り歩数 平均 ${f1(avg(firstWall.map((m) => m.opDist)))}、自分の残り歩数 平均 ${f1(avg(firstWall.map((m) => m.myDist)))}`);
const opDistBuckets = new Map<number, { wall: number; total: number }>();
for (const m of all) {
  if (m.myWalls === 0) continue;
  const b = Math.min(m.opDist, 8);
  const e = opDistBuckets.get(b) ?? { wall: 0, total: 0 };
  e.total++;
  if (m.type === 'wall') e.wall++;
  opDistBuckets.set(b, e);
}
out('相手の残り歩数ごとの「壁を置いた割合」（壁が残っている手番のみ）:');
for (const [b, e] of [...opDistBuckets].sort((x, y) => x[0] - y[0])) {
  out(`  相手あと ${b}${b === 8 ? '+' : ' '}歩: ${pct(e.wall, e.total)}（${e.total} 手番）`);
}

// ---- 形勢ごとの判断 ----
out('\n## 形勢（歩数の差）ごとの判断（壁が残っている手番のみ）');
out('  形勢 = 相手の残り歩数 - 自分の残り歩数（＋なら自分がリード。同じなら手番の自分が 1 歩早い）');
const lead = new Map<number, { wall: number; total: number; won: number }>();
for (const m of all) {
  if (m.myWalls === 0) continue;
  const d = Math.max(-4, Math.min(4, m.opDist - m.myDist));
  const e = lead.get(d) ?? { wall: 0, total: 0, won: 0 };
  e.total++;
  if (m.type === 'wall') e.wall++;
  if (m.won) e.won++;
  lead.set(d, e);
}
for (const [d, e] of [...lead].sort((x, y) => x[0] - y[0])) {
  out(`  形勢 ${d > 0 ? '+' : ''}${d}${Math.abs(d) === 4 ? '以上' : '    '}: 壁を置いた ${pct(e.wall, e.total)} / その後勝った ${pct(e.won, e.total)}（${e.total} 手番）`);
}

// ---- 壁の効き目 ----
out('\n## 置いた壁の効き目');
const walls = all.filter((m) => m.type === 'wall');
const gainDist = new Map<number, number>();
for (const m of walls) {
  const g = Math.min(m.opDistAfter - m.opDist, 4);
  gainDist.set(g, (gainDist.get(g) ?? 0) + 1);
}
out(`相手の道を伸ばした歩数: ${[...gainDist].sort((x, y) => x[0] - y[0]).map(([g, n]) => `${g}${g === 4 ? '+' : ''}歩 ${pct(n, walls.length).trim()}`).join(' / ')}`);
out(`自分の道を伸ばしてしまった壁: ${pct(walls.filter((m) => m.myDistAfter > m.myDist).length, walls.length).trim()}`);
const kinds = ['block', 'near', 'guard'] as const;
const kindName = { block: '相手の最短経路をふさぐ', near: 'コマに接する', guard: '自分の経路沿いの守り' };
out(`壁の種類: ${kinds.map((k) => `${kindName[k]} ${pct(walls.filter((m) => m.kind === k).length, walls.length).trim()}`).join(' / ')}`);
out(`勝者の壁の使用数 平均 ${f1(avg(records.filter((g) => g.winner !== null).map((g) => g.moves.filter((m) => m.player === g.winner && m.type === 'wall').length)))}、敗者 平均 ${f1(avg(records.filter((g) => g.winner !== null).map((g) => g.moves.filter((m) => m.player !== g.winner && m.type === 'wall').length)))}`);

// ---- コマの動かし方 ----
out('\n## コマの動かし方');
const pawns = all.filter((m) => m.type === 'pawn');
out(`最短経路に沿って前進: ${pct(pawns.filter((m) => m.myDistAfter < m.myDist).length, pawns.length).trim()} / 歩数が変わらない移動: ${pct(pawns.filter((m) => m.myDistAfter === m.myDist).length, pawns.length).trim()} / 遠回り: ${pct(pawns.filter((m) => m.myDistAfter > m.myDist).length, pawns.length).trim()}`);

// ---- 壁を先に使い切った側 ----
out('\n## 壁を先に使い切った側');
let firstOut = 0;
let firstOutWon = 0;
for (const g of records) {
  if (g.winner === null) continue;
  const m = g.moves.find((x) => x.type === 'wall' && x.myWalls === 1);
  if (!m) continue;
  firstOut++;
  if (g.winner === m.player) firstOutWon++;
}
out(`先に壁を使い切った側の勝率: ${pct(firstOutWon, firstOut)}（${firstOut} 局）`);

// ---- 歩数勝負の予測 ----
out('\n## 壁が尽きた後の歩数勝負');
let locked = 0;
let predicted = 0;
for (const g of records) {
  if (g.winner === null) continue;
  // 両者の壁が 0 になった最初の手番で、歩数を数えるだけで勝者が分かるか
  const m = g.moves.find((x) => x.myWalls === 0 && x.opWalls === 0);
  if (!m) continue;
  locked++;
  const moverWins = m.myDist <= m.opDist;
  if ((g.winner === m.player) === moverWins) predicted++;
}
out(`両者の壁が尽きた局面で「手番側は同じ歩数なら先着」の数え方で勝者を当てられた割合: ${pct(predicted, locked)}（${locked} 局）`);

// ---- 序盤 ----
out('\n## 序盤の手（最初の 4 手）');
const firstTen = new Set(records.map((g) => g.moves.slice(0, 10).map((m) => m.move).join(' ')));
out(`最初の 10 手が異なる対局: ${firstTen.size} 通り / ${records.length} 局（序盤 ${OPENING_PLIES} 手は最善から 1 歩分以内の手をランダムに選ぶ）`);
const openings = new Map<string, number>();
for (const g of records) {
  const k = g.moves.slice(0, 4).map((m) => m.move).join(' ');
  openings.set(k, (openings.get(k) ?? 0) + 1);
}
for (const [k, n] of [...openings].sort((x, y) => y[1] - x[1]).slice(0, 5)) out(`  ${k}: ${pct(n, records.length).trim()}`);

console.log(lines.join('\n'));
