// 手引き AI と探索 AI の不一致について、両方の候補手から探索 AI 同士で終局まで指し継ぐ。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/counterfactual.ts --games 120 --top 30 --depth 3 --rollout-depth 3

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { moveFromString } from '../../src/engine/game';
import type { CounterfactualResult } from '../../src/guide/counterfactual';
import type { Disagreement } from '../../src/guide/discover';
import { CounterfactualPool, DiscoveryPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '120' },
    top: { type: 'string', default: '30' },
    depth: { type: 'string', default: '3' },
    'rollout-depth': { type: 'string', default: '3' },
    kind: { type: 'string', default: 'all' },
  },
});

const champion = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'));
const games = Number(args.games);
const top = Number(args.top);
const depth = Number(args.depth);
const rolloutDepth = Number(args['rollout-depth']);
const requestedKind = args.kind;

const discoveryPool = new DiscoveryPool();
const gamesFound = await discoveryPool.runAll(
  Array.from({ length: games }, (_, i) => ({
    weights: champion.weights,
    depth,
    guidePlayer: (i % 2) as 0 | 1,
    seed: 51_000_000 + i,
  })),
);
await discoveryPool.close();

const positionKey = (d: Disagreement) =>
  JSON.stringify([d.position.pawns, d.position.wallsLeft, d.position.hWalls, d.position.vWalls, d.position.guidePlayer]);
const unique = new Map<string, Disagreement>();
for (const disagreement of gamesFound.flatMap((game) => game.disagreements)) {
  if (disagreement.guideWon || disagreement.regret < 1) continue;
  const disagreementKind = `${disagreement.guideType}-${disagreement.expertType}`;
  if (requestedKind !== 'all' && requestedKind !== disagreementKind) continue;
  const key = positionKey(disagreement);
  const previous = unique.get(key);
  if (!previous || disagreement.regret > previous.regret) unique.set(key, disagreement);
}
const selected = [...unique.values()].sort((a, b) => b.regret - a.regret).slice(0, top);

const counterfactualPool = new CounterfactualPool();
const results = await counterfactualPool.runAll(
  selected.map((disagreement) => ({ disagreement, weights: champion.weights, depth: rolloutDepth })),
);
await counterfactualPool.close();

const score = (result: CounterfactualResult, branch: 'guide' | 'expert') => {
  const winner = result[branch].winner;
  return winner === null ? 0.5 : winner === result.disagreement.position.guidePlayer ? 1 : 0;
};
const resultName = (result: CounterfactualResult, branch: 'guide' | 'expert') => {
  const value = score(result, branch);
  return value === 1 ? '勝ち' : value === 0.5 ? '引分' : '負け';
};
const kind = (result: CounterfactualResult) => `${result.disagreement.guideType} → ${result.disagreement.expertType}`;
const wallSequence = (result: CounterfactualResult, branch: 'guide' | 'expert', player: 'me' | 'opponent') =>
  result[branch].moves.filter((text, index) => {
    const move = moveFromString(text);
    const isMyTurn = index % 2 === 0;
    return move?.type === 'wall' && (player === 'me' ? isMyTurn : !isMyTurn);
  });
const nextOwnWall = (result: CounterfactualResult, branch: 'guide' | 'expert') => {
  for (let index = 2; index < result[branch].moves.length; index += 2) {
    const move = moveFromString(result[branch].moves[index]);
    if (move?.type === 'wall') return { move, text: result[branch].moves[index], delay: index };
  }
  return null;
};
const wallAnchorDistance = (aText: string, bText: string) => {
  const a = moveFromString(aText);
  const b = moveFromString(bText);
  if (a?.type !== 'wall' || b?.type !== 'wall') return null;
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
};

console.log(`# 反実仮想分析: 手引き AI 対 第 ${champion.generation} 世代`);
console.log();
console.log(`発見対局: ${games} 局（${depth} 手読み）`);
console.log(`終局追跡: 敗戦中・推定損失1点以上の上位 ${results.length} 局面（${rolloutDepth} 手読み）`);
console.log(`不一致型: ${requestedKind}`);
console.log('各局面で最初の候補手だけを変え、その後は同じ探索 AI 同士で終局まで指しています。');

const improved = results.filter((result) => score(result, 'expert') > score(result, 'guide'));
const worsened = results.filter((result) => score(result, 'expert') < score(result, 'guide'));
const unchanged = results.filter((result) => score(result, 'expert') === score(result, 'guide'));
console.log('\n## 勝敗への影響');
console.log(`- 探索AIの手で改善: ${improved.length} / ${results.length}`);
console.log(`- 変化なし: ${unchanged.length} / ${results.length}`);
console.log(`- 探索AIの手で悪化: ${worsened.length} / ${results.length}`);

const kinds = [...new Set(results.map(kind))];
console.log('\n## 不一致の種類別');
for (const name of kinds) {
  const group = results.filter((result) => kind(result) === name);
  const better = group.filter((result) => score(result, 'expert') > score(result, 'guide')).length;
  const same = group.filter((result) => score(result, 'expert') === score(result, 'guide')).length;
  const worse = group.length - better - same;
  console.log(`- ${name}: 改善 ${better} / 同じ ${same} / 悪化 ${worse}（計 ${group.length}）`);
}

const average = (values: number[]) => (values.length ? (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2) : '-');
const wallGroups = [
  ['改善', improved.filter((result) => result.disagreement.wallComparison)],
  ['同じ・悪化', [...unchanged, ...worsened].filter((result) => result.disagreement.wallComparison)],
] as const;
if (wallGroups.some(([, group]) => group.length)) {
  console.log('\n## 壁 → 壁の勝敗別特徴');
  for (const [name, group] of wallGroups) {
    const disagreements = group.map((result) => result.disagreement);
    console.log(`- ${name}（${group.length}局面）:`);
    console.log(
      `  - 残り歩数平均: 自分 ${average(disagreements.map((d) => d.myDist))} / 相手 ${average(disagreements.map((d) => d.opDist))}`,
    );
    console.log(
      `  - 手引き壁: 正味 ${average(disagreements.map((d) => d.wallComparison!.guide.netGain))} / 次の壁脅威差 ${average(disagreements.map((d) => d.wallComparison!.guide.myThreat - d.wallComparison!.guide.opThreat))}`,
    );
    console.log(
      `  - 探索AI壁: 正味 ${average(disagreements.map((d) => d.wallComparison!.expert.netGain))} / 次の壁脅威差 ${average(disagreements.map((d) => d.wallComparison!.expert.myThreat - d.wallComparison!.expert.opThreat))}`,
    );
    console.log(
      `  - 探索AI壁による差: 正味 ${average(disagreements.map((d) => d.wallComparison!.expert.netGain - d.wallComparison!.guide.netGain))} / 次の壁脅威 ${average(disagreements.map((d) => d.wallComparison!.expert.myThreat - d.wallComparison!.expert.opThreat - (d.wallComparison!.guide.myThreat - d.wallComparison!.guide.opThreat)))}`,
    );
  }
  console.log('- 条件別の内訳（改善 / 同じ・悪化）:');
  const conditions = [
    ['自分が残り5歩以内', (d: Disagreement) => d.myDist <= 5],
    ['歩数で先着可能', (d: Disagreement) => d.opDist - d.myDist >= 0],
    ['探索AI壁の次の壁脅威が2歩以上良い', (d: Disagreement) => {
      const wall = d.wallComparison!;
      return wall.expert.myThreat - wall.expert.opThreat - (wall.guide.myThreat - wall.guide.opThreat) >= 2;
    }],
    [
      '残り5歩以内 × 先着可能 × 次の壁脅威が2歩以上良い',
      (d: Disagreement) => {
        const wall = d.wallComparison!;
        return (
          d.myDist <= 5 &&
          d.opDist - d.myDist >= 0 &&
          wall.expert.myThreat - wall.expert.opThreat - (wall.guide.myThreat - wall.guide.opThreat) >= 2
        );
      },
    ],
  ] as const;
  for (const [name, includes] of conditions) {
    const better = wallGroups[0][1].filter((result) => includes(result.disagreement)).length;
    const other = wallGroups[1][1].filter((result) => includes(result.disagreement)).length;
    console.log(`  - ${name}: ${better} / ${other}`);
  }

  const focused = results.filter((result) => conditions.at(-1)![1](result.disagreement));
  console.log('\n## 終盤で先着可能かつ次の壁脅威が改善する局面');
  console.log(`- 対象: ${focused.length} 局面`);
  const focusedImproved = focused.filter((result) => score(result, 'expert') > score(result, 'guide'));
  const focusedOther = focused.filter((result) => score(result, 'expert') <= score(result, 'guide'));
  for (const [name, group] of [
    ['改善', focusedImproved],
    ['非改善', focusedOther],
  ] as const) {
    const nextWalls = group.map((result) => nextOwnWall(result, 'expert')).filter((wall) => wall !== null);
    const distances = group
      .map((result) => {
        const next = nextOwnWall(result, 'expert');
        return next ? wallAnchorDistance(result.disagreement.expertMove, next.text) : null;
      })
      .filter((distance) => distance !== null);
    console.log(
      `- ${name}: ${group.length}局面 / 次の自分の壁あり ${nextWalls.length} / 平均 ${average(nextWalls.map((wall) => wall.delay))}手後 / アンカー距離 ${average(distances)}`,
    );
  }
  for (const [index, result] of focused.entries()) {
    const d = result.disagreement;
    const next = nextOwnWall(result, 'expert');
    console.log(`\n### F${index + 1}. ${score(result, 'expert') > score(result, 'guide') ? '改善' : '非改善'}・${d.position.ply}手目`);
    console.log(`- 最初の壁: 手引き ${d.guideMove} / 探索AI ${d.expertMove}`);
    console.log(
      `- 探索AI側の壁列: 自分 [${wallSequence(result, 'expert', 'me').join(' ')}] / 相手 [${wallSequence(result, 'expert', 'opponent').join(' ')}]`,
    );
    console.log(
      `- 次の自分の壁: ${next ? `${next.text}（${next.delay}手後、アンカー距離${wallAnchorDistance(d.expertMove, next.text)}、向き${d.expertMove.at(-1) === next.text.at(-1) ? '同じ' : '変更'}）` : 'なし'}`,
    );
    console.log(`- 探索AI側の最初の16手: ${result.expert.moves.slice(0, 16).join(' ')}`);
  }
}

console.log('\n## 勝敗が変わった局面');
const changed = results.filter((result) => score(result, 'guide') !== score(result, 'expert'));
if (changed.length === 0) console.log('- なし');
for (const [i, result] of changed.entries()) {
  const d = result.disagreement;
  console.log(`\n### ${i + 1}. ${d.position.ply} 手目・${kind(result)}・regret ${d.regret}`);
  console.log(`- 手引き ${d.guideMove}: ${resultName(result, 'guide')}（残り ${result.guide.remainingPlies} 手）`);
  console.log(`- 探索AI ${d.expertMove}: ${resultName(result, 'expert')}（残り ${result.expert.remainingPlies} 手）`);
  console.log(`- 残り歩数: 自分 ${d.myDist} / 相手 ${d.opDist}、壁: 自分 ${d.myWalls} / 相手 ${d.opWalls}`);
  if (d.wallComparison) {
    const guide = d.wallComparison.guide;
    const expert = d.wallComparison.expert;
    console.log(
      `- 壁効果: 手引き 正味${guide.netGain}・脅威差${guide.myThreat - guide.opThreat} / 探索AI 正味${expert.netGain}・脅威差${expert.myThreat - expert.opThreat}`,
    );
  }
  console.log(`- 盤面: 駒 ${d.position.pawns.join(' / ')}、横壁 [${d.position.hWalls.join(' ')}]、縦壁 [${d.position.vWalls.join(' ')}]`);
  console.log(`- 直前までの棋譜: ${d.position.moves || '（初期局面）'}`);
}
