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
const uniquePositions = (items: Disagreement[]) => {
  const unique = new Map<string, Disagreement>();
  for (const d of items) {
    const key = JSON.stringify([
      d.position.pawns,
      d.position.wallsLeft,
      d.position.hWalls,
      d.position.vWalls,
      d.position.guidePlayer,
    ]);
    const previous = unique.get(key);
    if (!previous || d.regret > previous.regret) unique.set(key, d);
  }
  return [...unique.values()];
};
const meaningful = uniquePositions(meaningfulRaw);
const winningRaw = all.filter((d) => d.guideWon && d.regret >= 1);
const winningMeaningful = uniquePositions(winningRaw);

console.log(`# 戦術発見用ログ: 手引き AI 対 第 ${champion.generation} 世代`);
console.log();
console.log(`対局: ${games} 局（${depth} 手読み） / 手引き AI 勝率 ${pct(won, games)} / 引き分け ${drawn} 局`);
console.log(
  `判断の不一致: ${all.length} 局面 / 手引き敗戦中 ${losses.length} 局面 / 推定損失 1 点以上 ${meaningfulRaw.length} 局面（重複除去後 ${meaningful.length}）`,
);
console.log(`対照群: 手引き勝利中の推定損失 1 点以上 ${winningRaw.length} 局面（重複除去後 ${winningMeaningful.length}）`);

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

const wallToPawn = meaningful.filter((d) => d.wallPawnComparison);
if (wallToPawn.length) {
  console.log('\n## 壁 → 前進の比較');
  console.log(`- 対象: ${wallToPawn.length} 局面`);
  const netBins = [
    ['0歩以下', (n: number) => n <= 0],
    ['1歩', (n: number) => n === 1],
    ['2〜3歩', (n: number) => n >= 2 && n <= 3],
    ['4歩以上', (n: number) => n >= 4],
  ] as const;
  for (const [name, includes] of netBins) {
    const at = wallToPawn.filter((d) => includes(d.wallPawnComparison!.guide.netGain));
    if (!at.length) continue;
    const regret = at.reduce((sum, d) => sum + d.regret, 0) / at.length;
    console.log(`- 壁の即時正味効果 ${name}: ${at.length} 局面、平均推定損失 ${regret.toFixed(1)} 点`);
  }
  const wallBetter = wallToPawn.filter(
    (d) => d.wallPawnComparison!.guide.netGain > d.wallPawnComparison!.expert.progress,
  ).length;
  const same = wallToPawn.filter(
    (d) => d.wallPawnComparison!.guide.netGain === d.wallPawnComparison!.expert.progress,
  ).length;
  const pawnBetter = wallToPawn.length - wallBetter - same;
  const lastWall = wallToPawn.filter((d) => d.myWalls === 1).length;
  const noRaceImprovement = wallToPawn.filter((d) => {
    const c = d.wallPawnComparison!;
    return d.opDist + c.guide.opDistanceGain - (d.myDist + c.guide.myDistanceCost) <= d.opDist - d.myDist;
  }).length;
  const wallCreatesLead = wallToPawn.filter((d) => {
    const c = d.wallPawnComparison!;
    const before = d.opDist - d.myDist;
    const after = d.opDist + c.guide.opDistanceGain - (d.myDist + c.guide.myDistanceCost);
    return before <= 0 && after > 0;
  }).length;
  const distanceBins = [
    ['自分が残り4歩以内', (d: Disagreement) => d.myDist <= 4],
    ['自分が残り5〜8歩', (d: Disagreement) => d.myDist >= 5 && d.myDist <= 8],
    ['自分が残り9歩以上', (d: Disagreement) => d.myDist >= 9],
  ] as const;
  for (const [name, includes] of distanceBins) {
    const at = wallToPawn.filter(includes);
    if (!at.length) continue;
    const regret = at.reduce((sum, d) => sum + d.regret, 0) / at.length;
    console.log(`- ${name}: ${at.length} 局面、平均推定損失 ${regret.toFixed(1)} 点`);
  }
  const cutStepBins = [
    ['相手の1歩先', (n: number | null) => n === 1],
    ['相手の2〜3歩先', (n: number | null) => n !== null && n >= 2 && n <= 3],
    ['相手の4〜6歩先', (n: number | null) => n !== null && n >= 4 && n <= 6],
    ['相手の7歩以上先', (n: number | null) => n !== null && n >= 7],
    ['現在の代表最短路以外', (n: number | null) => n === null],
  ] as const;
  for (const [name, includes] of cutStepBins) {
    const at = wallToPawn.filter((d) => includes(d.wallPawnComparison!.guide.routeCutStep));
    if (!at.length) continue;
    const regret = at.reduce((sum, d) => sum + d.regret, 0) / at.length;
    const netGain = at.reduce((sum, d) => sum + d.wallPawnComparison!.guide.netGain, 0) / at.length;
    const replyPawn = at.filter((d) => d.wallPawnComparison!.guideAfterReply.replyType === 'pawn').length;
    console.log(
      `- 壁の位置 ${name}: ${at.length} 局面、平均推定損失 ${regret.toFixed(1)} 点、平均正味効果 ${netGain.toFixed(1)} 歩、相手が前進 ${replyPawn}`,
    );
  }
  const leadBins = [
    ['歩数で負け・同点', (d: Disagreement) => d.opDist - d.myDist <= 0],
    ['1〜3歩リード', (d: Disagreement) => d.opDist - d.myDist >= 1 && d.opDist - d.myDist <= 3],
    ['4歩以上リード', (d: Disagreement) => d.opDist - d.myDist >= 4],
  ] as const;
  for (const [name, includes] of leadBins) {
    const at = wallToPawn.filter(includes);
    if (!at.length) continue;
    const regret = at.reduce((sum, d) => sum + d.regret, 0) / at.length;
    console.log(`- ${name}: ${at.length} 局面、平均推定損失 ${regret.toFixed(1)} 点`);
  }
  const threatDeltas = wallToPawn.map(
    (d) => d.wallPawnComparison!.expert.myThreat - d.wallPawnComparison!.guide.myThreat,
  );
  const pawnPreservesMoreThreat = threatDeltas.filter((n) => n > 0).length;
  const averageThreatDelta = threatDeltas.reduce((a, b) => a + b, 0) / threatDeltas.length;
  const replyTypes = (branch: 'guideAfterReply' | 'expertAfterReply') => {
    const counts = new Map<string, number>();
    for (const d of wallToPawn) {
      const reply = d.wallPawnComparison![branch].replyType ?? 'なし（決着）';
      counts.set(reply, (counts.get(reply) ?? 0) + 1);
    }
    return [...counts].map(([type, n]) => `${type} ${n}`).join(' / ');
  };
  const replyRaceDeltas = wallToPawn.map(
    (d) => d.wallPawnComparison!.expertAfterReply.raceLead - d.wallPawnComparison!.guideAfterReply.raceLead,
  );
  const pawnLineBetterAfterReply = replyRaceDeltas.filter((n) => n > 0).length;
  const sameAfterReply = replyRaceDeltas.filter((n) => n === 0).length;
  const wallLineBetterAfterReply = replyRaceDeltas.filter((n) => n < 0).length;
  const averageRaceDelta = replyRaceDeltas.reduce((a, b) => a + b, 0) / replyRaceDeltas.length;
  const secureRace = (branch: 'guideAfterReply' | 'expertAfterReply') =>
    wallToPawn.filter((d) => {
      const c = d.wallPawnComparison![branch];
      return c.opWalls === 0 && c.myDist <= c.opDist;
    }).length;
  const opponentWins = (branch: 'guideAfterReply' | 'expertAfterReply') =>
    wallToPawn.filter((d) => d.wallPawnComparison![branch].winner === (1 - d.position.guidePlayer)).length;
  const wallGainErased = wallToPawn.filter((d) => {
    const c = d.wallPawnComparison!;
    const immediateLead = d.opDist + c.guide.opDistanceGain - (d.myDist + c.guide.myDistanceCost);
    return c.guideAfterReply.raceLead < immediateLead;
  }).length;
  const replyPatterns = new Map<string, { n: number; regret: number }>();
  for (const d of wallToPawn) {
    const c = d.wallPawnComparison!;
    const key = `壁後 ${c.guideAfterReply.replyType ?? '決着'} / 前進後 ${c.expertAfterReply.replyType ?? '決着'}`;
    const value = replyPatterns.get(key) ?? { n: 0, regret: 0 };
    value.n++;
    value.regret += d.regret;
    replyPatterns.set(key, value);
  }
  const initiativeSwing = wallToPawn.filter((d) => {
    const c = d.wallPawnComparison!;
    return c.guideAfterReply.replyType === 'pawn' && c.expertAfterReply.replyType === 'wall';
  });
  const initiativeNearGoal = initiativeSwing.filter((d) => d.myDist <= 4).length;
  const initiativeRegret = initiativeSwing.reduce((sum, d) => sum + d.regret, 0) / initiativeSwing.length;
  console.log(`- 即時効果と探索AIの前進量: 壁が大きい ${wallBetter} / 同じ ${same} / 前進が大きい ${pawnBetter}`);
  console.log(`- 最後の壁を使う: ${lastWall} 局面`);
  console.log(`- 壁を置いても歩数リードが増えない: ${noRaceImprovement} 局面`);
  console.log(`- 壁で歩数上の不利・同点をリードへ変える: ${wallCreatesLead} 局面`);
  console.log(`- 前進後のほうが次の壁脅威を多く残す: ${pawnPreservesMoreThreat} 局面、平均差 ${averageThreatDelta.toFixed(2)} 歩`);
  console.log(`- 壁手への最善応手: ${replyTypes('guideAfterReply')}`);
  console.log(`- 前進手への最善応手: ${replyTypes('expertAfterReply')}`);
  console.log(
    `- 最善応手後の歩数リード: 前進側が良い ${pawnLineBetterAfterReply} / 同じ ${sameAfterReply} / 壁側が良い ${wallLineBetterAfterReply} / 前進側の平均差 ${averageRaceDelta.toFixed(2)} 歩`,
  );
  console.log(`- 相手の壁が尽き、先着できる形: 壁側 ${secureRace('guideAfterReply')} / 前進側 ${secureRace('expertAfterReply')}`);
  console.log(`- 相手の最善応手までに敗着: 壁側 ${opponentWins('guideAfterReply')} / 前進側 ${opponentWins('expertAfterReply')}`);
  console.log(`- 壁の即時リードが相手の最善応手で縮む: ${wallGainErased} 局面`);
  console.log('- 応手の組み合わせ:');
  for (const [pattern, value] of [...replyPatterns].sort((a, b) => b[1].n - a[1].n)) {
    console.log(`  - ${pattern}: ${value.n} 局面、平均推定損失 ${(value.regret / value.n).toFixed(1)} 点`);
  }
  if (initiativeSwing.length) {
    console.log(
      `- 主導権の逆転（自分が壁なら相手は前進、自分が前進なら相手は壁）: ${initiativeSwing.length} 局面（自分が残り4歩以内 ${initiativeNearGoal}）、平均推定損失 ${initiativeRegret.toFixed(1)} 点`,
    );
  }

  const winningWallToPawn = winningMeaningful.filter((d) => d.wallPawnComparison);
  if (winningWallToPawn.length) {
    console.log('\n## 壁 → 前進の勝敗対照');
    console.log(`- 敗戦中: ${wallToPawn.length} 局面 / 勝利中: ${winningWallToPawn.length} 局面`);
    const contrast = (name: string, includes: (d: Disagreement) => boolean) => {
      const losing = wallToPawn.filter(includes).length;
      const winning = winningWallToPawn.filter(includes).length;
      const losingRate = (losing / wallToPawn.length) * 100;
      const winningRate = (winning / winningWallToPawn.length) * 100;
      console.log(
        `- ${name}: 敗戦 ${losing}（${losingRate.toFixed(1)}%）/ 勝利 ${winning}（${winningRate.toFixed(1)}%）/ 差 ${(losingRate - winningRate).toFixed(1)}pt`,
      );
    };
    for (const [name, includes] of cutStepBins) {
      contrast(`壁の位置 ${name}`, (d) => includes(d.wallPawnComparison!.guide.routeCutStep));
    }
    for (const [name, includes] of netBins) {
      contrast(`壁の即時正味効果 ${name}`, (d) => includes(d.wallPawnComparison!.guide.netGain));
    }
    for (const [name, includes] of leadBins) contrast(name, includes);
    contrast('自分が残り4歩以内', (d) => d.myDist <= 4);
    contrast('最後の壁を使う', (d) => d.myWalls === 1);
    contrast('壁手へ相手が前進で返す', (d) => d.wallPawnComparison!.guideAfterReply.replyType === 'pawn');
    contrast('前進手へ相手が壁で返す', (d) => d.wallPawnComparison!.expertAfterReply.replyType === 'wall');
    contrast('壁なら相手が前進・前進なら相手が壁', (d) => {
      const c = d.wallPawnComparison!;
      return c.guideAfterReply.replyType === 'pawn' && c.expertAfterReply.replyType === 'wall';
    });
  }

  console.log(`\n## 壁 → 前進の推定損失上位 ${Math.min(top, wallToPawn.length)} 局面`);
  for (const [i, d] of [...wallToPawn].sort((a, b) => b.regret - a.regret).slice(0, top).entries()) {
    const w = d.wallPawnComparison!.guide;
    const p = d.wallPawnComparison!.expert;
    console.log(`\n### W${i + 1}. ${phase(d)}・${d.position.ply} 手目・regret ${d.regret}`);
    console.log(`- 手引き: ${d.guideMove}（相手 +${w.opDistanceGain} / 自分 +${w.myDistanceCost} / 正味 ${w.netGain}歩）`);
    console.log(`- 壁の位置: 相手の代表最短路の ${w.routeCutStep ?? '外'} 歩先、相手駒から壁周辺まで ${w.pawnDistanceToWall} マス`);
    console.log(`- 探索AI: ${d.expertMove}（前進 ${p.progress}、戻る ${p.backtrack ? 'はい' : 'いいえ'}）`);
    console.log(`- 壁後: 壁脅威 自分 ${w.myThreat} / 相手 ${w.opThreat}、最短ルート多様性 自分 ${w.myRoutes} / 相手 ${w.opRoutes}`);
    console.log(`- 前進後: 壁脅威 自分 ${p.myThreat} / 相手 ${p.opThreat}、最短ルート多様性 自分 ${p.myRoutes} / 相手 ${p.opRoutes}`);
    const gw = d.wallPawnComparison!.guideAfterReply;
    const ep = d.wallPawnComparison!.expertAfterReply;
    console.log(`- 壁手への最善応手: ${gw.reply ?? '（決着）'}、応手後の残り歩数 自分 ${gw.myDist} / 相手 ${gw.opDist}、リード ${gw.raceLead}`);
    console.log(`- 前進手への最善応手: ${ep.reply ?? '（決着）'}、応手後の残り歩数 自分 ${ep.myDist} / 相手 ${ep.opDist}、リード ${ep.raceLead}`);
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
