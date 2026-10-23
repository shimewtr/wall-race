// 手引き AI が最新チャンピオンと違う手を選ぶ局面を集め、LLM で読める形に要約する。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/discover.ts --games 120 --top 30

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import type { Disagreement } from '../../src/guide/discover';
import { DiscoveryPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '120' },
    depth: { type: 'string', default: '3' },
    top: { type: 'string', default: '30' },
  },
});

const champion = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'));
const games = Number(args.games);
const depth = Number(args.depth);
const top = Number(args.top);
const pool = new DiscoveryPool();
const records = await pool.runAll(
  Array.from({ length: games }, (_, i) => ({
    weights: champion.weights,
    depth,
    guidePlayer: (i % 2) as 0 | 1,
    seed: 41_000_000 + i,
  })),
);
await pool.close();

const pct = (n: number, d: number) => (d ? `${((n / d) * 100).toFixed(1)}%` : '-');
const won = records.filter((g) => g.winner === g.guidePlayer).length;
const drawn = records.filter((g) => g.winner === null).length;
const all = records.flatMap((g) => g.disagreements);
const losses = all.filter((d) => !d.guideWon);
const meaningfulRaw = losses.filter((d) => d.regret >= 1);
// 序盤の乱数だけが違って同じ盤面へ合流することがある。LLM に同じ例を何度も見せない。
const unique = new Map<string, Disagreement>();
for (const d of meaningfulRaw) {
  const key = JSON.stringify([d.position.pawns, d.position.wallsLeft, d.position.hWalls, d.position.vWalls, d.position.guidePlayer]);
  const previous = unique.get(key);
  if (!previous || d.regret > previous.regret) unique.set(key, d);
}
const meaningful = [...unique.values()];

console.log(`# 戦術発見用ログ: 手引き AI 対 第 ${champion.generation} 世代`);
console.log();
console.log(`対局: ${games} 局（${depth} 手読み） / 手引き AI 勝率 ${pct(won, games)} / 引き分け ${drawn} 局`);
console.log(
  `判断の不一致: ${all.length} 局面 / 手引き敗戦中 ${losses.length} 局面 / 推定損失 1 点以上 ${meaningfulRaw.length} 局面（重複除去後 ${meaningful.length}）`,
);

const kinds = new Map<string, { n: number; regret: number }>();
for (const d of meaningful) {
  const key = `${d.guideType} → ${d.expertType}`;
  const e = kinds.get(key) ?? { n: 0, regret: 0 };
  e.n++;
  e.regret += d.regret;
  kinds.set(key, e);
}
console.log('\n## 不一致の種類');
for (const [kind, e] of [...kinds].sort((a, b) => b[1].n - a[1].n)) {
  console.log(`- ${kind}: ${e.n} 局面、平均推定損失 ${(e.regret / e.n).toFixed(1)} 点`);
}

const phase = (d: Disagreement) => (d.position.ply < 12 ? '序盤' : d.myWalls + d.opWalls <= 6 ? '終盤' : '中盤');
const wallToWall = meaningful.filter((d) => d.wallComparison);
if (wallToWall.length) {
  const orientation = new Map<string, number>();
  for (const d of wallToWall) {
    const key = `${d.guideMove.at(-1)} → ${d.expertMove.at(-1)}`;
    orientation.set(key, (orientation.get(key) ?? 0) + 1);
  }
  console.log('\n## 壁 → 壁の比較');
  console.log(`- 対象: ${wallToWall.length} 局面`);
  console.log(`- 向き: ${[...orientation].map(([k, n]) => `${k} ${n}`).join('、')}`);
  const metrics = [
    ['相手を延ばす歩数', (d: Disagreement) => d.wallComparison!.expert.opDistanceGain - d.wallComparison!.guide.opDistanceGain],
    ['自分を延ばす歩数（少ないほど良い）', (d: Disagreement) => d.wallComparison!.guide.myDistanceCost - d.wallComparison!.expert.myDistanceCost],
    ['正味の遠回り', (d: Disagreement) => d.wallComparison!.expert.netGain - d.wallComparison!.guide.netGain],
    [
      '壁脅威の差',
      (d: Disagreement) =>
        d.wallComparison!.expert.myThreat -
        d.wallComparison!.expert.opThreat -
        (d.wallComparison!.guide.myThreat - d.wallComparison!.guide.opThreat),
    ],
    [
      '最短ルート多様性の差',
      (d: Disagreement) =>
        d.wallComparison!.expert.myRoutes -
        d.wallComparison!.expert.opRoutes -
        (d.wallComparison!.guide.myRoutes - d.wallComparison!.guide.opRoutes),
    ],
  ] as const;
  for (const [name, delta] of metrics) {
    const values = wallToWall.map(delta);
    const better = values.filter((v) => v > 0).length;
    const same = values.filter((v) => v === 0).length;
    const average = values.reduce((a, b) => a + b, 0) / values.length;
    console.log(`- ${name}: 探索AIが良い ${better} / 同じ ${same} / 平均差 ${average.toFixed(2)}`);
  }
}

const pawnToPawn = meaningful.filter((d) => d.pawnComparison);
if (pawnToPawn.length) {
  console.log('\n## 前進 → 前進の比較');
  console.log(`- 対象: ${pawnToPawn.length} 局面`);
  const metrics = [
    ['その手で縮める歩数', (d: Disagreement) => d.pawnComparison!.expert.progress - d.pawnComparison!.guide.progress],
    [
      '中央への近さ',
      (d: Disagreement) => d.pawnComparison!.guide.centerOffset - d.pawnComparison!.expert.centerOffset,
    ],
    [
      '自分の次の壁脅威',
      (d: Disagreement) => d.pawnComparison!.expert.myThreat - d.pawnComparison!.guide.myThreat,
    ],
    [
      '相手から受ける壁脅威（少ないほど良い）',
      (d: Disagreement) => d.pawnComparison!.guide.opThreat - d.pawnComparison!.expert.opThreat,
    ],
    [
      '自分の最短ルート多様性',
      (d: Disagreement) => d.pawnComparison!.expert.myRoutes - d.pawnComparison!.guide.myRoutes,
    ],
  ] as const;
  for (const [name, delta] of metrics) {
    const values = pawnToPawn.map(delta);
    const better = values.filter((v) => v > 0).length;
    const worse = values.filter((v) => v < 0).length;
    const average = values.reduce((a, b) => a + b, 0) / values.length;
    console.log(`- ${name}: 探索AIが良い ${better} / 悪い ${worse} / 平均差 ${average.toFixed(2)}`);
  }
  const guideBacktracks = pawnToPawn.filter((d) => d.pawnComparison!.guide.backtrack).length;
  const expertBacktracks = pawnToPawn.filter((d) => d.pawnComparison!.expert.backtrack).length;
  const guideAdjacent = pawnToPawn.filter((d) => d.pawnComparison!.guide.adjacent).length;
  const expertAdjacent = pawnToPawn.filter((d) => d.pawnComparison!.expert.adjacent).length;
  console.log(`- 直近マスへ戻る: 手引き ${guideBacktracks} / 探索AI ${expertBacktracks}`);
  console.log(`- 相手と隣接する: 手引き ${guideAdjacent} / 探索AI ${expertAdjacent}`);

  console.log(`\n## 前進 → 前進の推定損失上位 ${Math.min(top, pawnToPawn.length)} 局面`);
  for (const [i, d] of [...pawnToPawn].sort((a, b) => b.regret - a.regret).slice(0, top).entries()) {
    const g = d.pawnComparison!.guide;
    const e = d.pawnComparison!.expert;
    console.log(`\n### P${i + 1}. ${phase(d)}・${d.position.ply} 手目・regret ${d.regret}`);
    console.log(`- 手引き: ${d.guideMove}（前進 ${g.progress}、中央距離 ${g.centerOffset}、戻る ${g.backtrack ? 'はい' : 'いいえ'}）`);
    console.log(`- 探索AI: ${d.expertMove}（前進 ${e.progress}、中央距離 ${e.centerOffset}、戻る ${e.backtrack ? 'はい' : 'いいえ'}）`);
    console.log(`- 手引き後: 壁脅威 自分 ${g.myThreat} / 相手 ${g.opThreat}、最短ルート多様性 自分 ${g.myRoutes} / 相手 ${g.opRoutes}`);
    console.log(`- 探索AI後: 壁脅威 自分 ${e.myThreat} / 相手 ${e.opThreat}、最短ルート多様性 自分 ${e.myRoutes} / 相手 ${e.opRoutes}`);
    console.log(`- 残り歩数: 自分 ${d.myDist} / 相手 ${d.opDist}、壁: 自分 ${d.myWalls} / 相手 ${d.opWalls}`);
    console.log(`- 盤面: 駒 ${d.position.pawns.join(' / ')}、横壁 [${d.position.hWalls.join(' ')}]、縦壁 [${d.position.vWalls.join(' ')}]`);
    console.log(`- 直前までの棋譜: ${d.position.moves || '（初期局面）'}`);
  }
}

const phases = new Map<string, number>();
for (const d of meaningful) phases.set(phase(d), (phases.get(phase(d)) ?? 0) + 1);
console.log('\n## 発生する局面');
for (const p of ['序盤', '中盤', '終盤']) console.log(`- ${p}: ${phases.get(p) ?? 0} 局面`);

console.log('\n## 自分の残り壁');
for (const n of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) {
  const at = meaningful.filter((d) => d.myWalls === n);
  if (at.length === 0) continue;
  const waste = at.filter((d) => d.guideType === 'wall' && d.expertType === 'pawn');
  console.log(`- ${n} 枚: ${at.length} 局面（手引きが壁・探索AIが前進: ${waste.length}）`);
}

console.log(`\n## 推定損失が大きい ${Math.min(top, meaningful.length)} 局面`);
console.log('`regret` は探索 AI の手と手引きの手の評価差。10 点がおよそ最短経路 1 歩分。');
for (const [i, d] of [...meaningful].sort((a, b) => b.regret - a.regret).slice(0, top).entries()) {
  console.log(`\n### ${i + 1}. ${phase(d)}・${d.position.ply} 手目・regret ${d.regret}`);
  console.log(`- 手引き: ${d.guideMove}（${d.guideType}、評価 ${d.guideScore}）`);
  console.log(`- 探索AI: ${d.expertMove}（${d.expertType}、評価 ${d.expertScore}）`);
  console.log(`- 残り歩数: 自分 ${d.myDist} / 相手 ${d.opDist}、壁: 自分 ${d.myWalls} / 相手 ${d.opWalls}`);
  console.log(`- 壁1枚の最大効果: 自分 ${d.myThreat} / 相手 ${d.opThreat}、最短ルート多様性: 自分 ${d.myRoutes} / 相手 ${d.opRoutes}`);
  console.log(`- 盤面: 駒 ${d.position.pawns.join(' / ')}、横壁 [${d.position.hWalls.join(' ')}]、縦壁 [${d.position.vWalls.join(' ')}]`);
  console.log(`- 直前までの棋譜: ${d.position.moves || '（初期局面）'}`);
}
