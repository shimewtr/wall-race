// 手引きと探索AIの不一致を終局まで追跡し、似た局面をクラスタ化する。
// 勝敗を改善するクラスタの代表例から、既存特徴量にない戦術候補を探す。
// 重い処理なのでGitHub ActionsのExperimentで実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import type { Weights } from '../../src/engine/ai';
import type { CounterfactualResult } from '../../src/guide/counterfactual';
import type { ContinuationEffects, Disagreement, PawnEffects, WallEffects } from '../../src/guide/discover';
import { CounterfactualPool, DiscoveryPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '240' },
    top: { type: 'string', default: '80' },
    clusters: { type: 'string', default: '5' },
    depth: { type: 'string', default: '3' },
    'rollout-depth': { type: 'string', default: '3' },
    seed: { type: 'string', default: '108000000' },
  },
});

const root = join(import.meta.dirname, '..', '..');
const champion = JSON.parse(readFileSync(join(root, 'training', 'champion.json'), 'utf8')) as {
  generation: number;
  weights: Weights;
};
const games = Number(args.games);
const top = Number(args.top);
const clusterCount = Number(args.clusters);
const depth = Number(args.depth);
const rolloutDepth = Number(args['rollout-depth']);
const seed = Number(args.seed);

const discoveryPool = new DiscoveryPool();
const discovered = await discoveryPool.runAll(Array.from({ length: games }, (_, index) => ({
  weights: champion.weights,
  depth,
  guidePlayer: (index % 2) as 0 | 1,
  seed: seed + index,
})));
await discoveryPool.close();

const positionKey = (row: Disagreement) => JSON.stringify([
  row.position.pawns,
  row.position.wallsLeft,
  row.position.hWalls,
  row.position.vWalls,
  row.position.guidePlayer,
]);
const unique = new Map<string, Disagreement>();
for (const row of discovered.flatMap((game) => game.disagreements)) {
  if (row.regret < 1) continue;
  const key = positionKey(row);
  const previous = unique.get(key);
  if (!previous || row.regret > previous.regret) unique.set(key, row);
}
const ranked = [...unique.values()].sort((a, b) => b.regret - a.regret);
const losing = ranked.filter((row) => !row.guideWon).slice(0, Math.ceil(top / 2));
const winning = ranked.filter((row) => row.guideWon).slice(0, Math.floor(top / 2));
const selected = [...losing, ...winning];
if (selected.length < top) {
  const used = new Set(selected.map(positionKey));
  selected.push(...ranked.filter((row) => !used.has(positionKey(row))).slice(0, top - selected.length));
}
if (!selected.length) throw new Error('クラスタ化できる不一致局面がありません');

const counterfactualPool = new CounterfactualPool();
const results = await counterfactualPool.runAll(selected.map((disagreement) => ({
  disagreement,
  weights: champion.weights,
  depth: rolloutDepth,
})));
await counterfactualPool.close();

const branchScore = (result: CounterfactualResult, branch: 'guide' | 'expert') => {
  const winner = result[branch].winner;
  return winner === null ? 0.5 : winner === result.disagreement.position.guidePlayer ? 1 : 0;
};
const outcomeDelta = (result: CounterfactualResult) => branchScore(result, 'expert') - branchScore(result, 'guide');
const balance = (value: { myThreat: number; opThreat: number }) => value.myThreat - value.opThreat;
const routeBalance = (value: { myRoutes: number; opRoutes: number }) => value.myRoutes - value.opRoutes;
const continuationDelta = (guide: ContinuationEffects, expert: ContinuationEffects) => ({
  replyRaceDelta: expert.raceLead - guide.raceLead,
  replyThreatDelta: balance(expert) - balance(guide),
  replyWallDelta: expert.myWalls - expert.opWalls - (guide.myWalls - guide.opWalls),
});
const wallDeltas = (guide: WallEffects, expert: WallEffects) => ({
  netGainDelta: expert.netGain - guide.netGain,
  threatDelta: balance(expert) - balance(guide),
  routeDelta: routeBalance(expert) - routeBalance(guide),
  extensionDelta: expert.extensionNetGain - guide.extensionNetGain,
  closerCut: (guide.routeCutStep ?? 10) - (expert.routeCutStep ?? 10),
  closerWall: guide.pawnDistanceToWall - expert.pawnDistanceToWall,
});
const pawnDeltas = (guide: PawnEffects, expert: PawnEffects) => ({
  pawnProgressDelta: expert.progress - guide.progress,
  threatDelta: balance(expert) - balance(guide),
  routeDelta: routeBalance(expert) - routeBalance(guide),
  backtrackAvoided: Number(guide.backtrack) - Number(expert.backtrack),
  centerGain: guide.centerOffset - expert.centerOffset,
});

const featureNames = [
  'ply', 'raceLead', 'wallDiff', 'totalWalls', 'threatLead', 'routeLead', 'regret',
  'pawnPawn', 'pawnWall', 'wallPawn', 'wallWall',
  'netGainDelta', 'pawnProgressDelta', 'threatDelta', 'routeDelta', 'extensionDelta',
  'closerCut', 'closerWall', 'backtrackAvoided', 'centerGain',
  'replyRaceDelta', 'replyThreatDelta', 'replyWallDelta',
] as const;
type FeatureName = typeof featureNames[number];
type FeatureRow = Record<FeatureName, number>;
const featureLabels: Record<FeatureName, string> = {
  ply: '手数', raceLead: '歩数リード', wallDiff: '壁枚数差', totalWalls: '盤上残り壁',
  threatLead: '壁脅威差', routeLead: '経路多様性差', regret: '探索上の損失',
  pawnPawn: '前進→前進', pawnWall: '前進→壁', wallPawn: '壁→前進', wallWall: '壁→壁',
  netGainDelta: '即時正味効果差', pawnProgressDelta: '前進量差', threatDelta: '手後の脅威差',
  routeDelta: '手後の経路差', extensionDelta: '連結2枚目効果差', closerCut: '相手に近い遮断',
  closerWall: '相手駒に近い壁', backtrackAvoided: '戻り回避', centerGain: '中央寄り',
  replyRaceDelta: '応手後の歩数差', replyThreatDelta: '応手後の脅威差', replyWallDelta: '応手後の壁差',
};

function features(disagreement: Disagreement): FeatureRow {
  const kind = `${disagreement.guideType}-${disagreement.expertType}`;
  const row = Object.fromEntries(featureNames.map((name) => [name, 0])) as FeatureRow;
  Object.assign(row, {
    ply: disagreement.position.ply,
    raceLead: disagreement.opDist - disagreement.myDist,
    wallDiff: disagreement.myWalls - disagreement.opWalls,
    totalWalls: disagreement.myWalls + disagreement.opWalls,
    threatLead: disagreement.myThreat - disagreement.opThreat,
    routeLead: disagreement.myRoutes - disagreement.opRoutes,
    regret: disagreement.regret,
    pawnPawn: Number(kind === 'pawn-pawn'),
    pawnWall: Number(kind === 'pawn-wall'),
    wallPawn: Number(kind === 'wall-pawn'),
    wallWall: Number(kind === 'wall-wall'),
  });
  if (disagreement.wallComparison) Object.assign(row, wallDeltas(disagreement.wallComparison.guide, disagreement.wallComparison.expert));
  if (disagreement.pawnComparison) Object.assign(row, pawnDeltas(disagreement.pawnComparison.guide, disagreement.pawnComparison.expert));
  if (disagreement.wallPawnComparison) {
    const comparison = disagreement.wallPawnComparison;
    row.netGainDelta = comparison.expert.progress - comparison.guide.netGain;
    row.threatDelta = balance(comparison.expert) - balance(comparison.guide);
    row.routeDelta = routeBalance(comparison.expert) - routeBalance(comparison.guide);
    row.backtrackAvoided = -Number(comparison.expert.backtrack);
    Object.assign(row, continuationDelta(comparison.guideAfterReply, comparison.expertAfterReply));
  }
  if (disagreement.pawnWallComparison) {
    const comparison = disagreement.pawnWallComparison;
    row.netGainDelta = comparison.expert.netGain - comparison.guide.progress;
    row.threatDelta = balance(comparison.expert) - balance(comparison.guide);
    row.routeDelta = routeBalance(comparison.expert) - routeBalance(comparison.guide);
    row.extensionDelta = comparison.expert.extensionNetGain;
    row.closerCut = 10 - (comparison.expert.routeCutStep ?? 10);
    row.closerWall = -comparison.expert.pawnDistanceToWall;
    Object.assign(row, continuationDelta(comparison.guideAfterReply, comparison.expertAfterReply));
  }
  return row;
}

const raw = results.map((result) => features(result.disagreement));
const means = Object.fromEntries(featureNames.map((name) => [name, raw.reduce((sum, row) => sum + row[name], 0) / raw.length])) as FeatureRow;
const deviations = Object.fromEntries(featureNames.map((name) => {
  const variance = raw.reduce((sum, row) => sum + (row[name] - means[name]) ** 2, 0) / raw.length;
  return [name, Math.sqrt(variance) || 1];
})) as FeatureRow;
const vectors = raw.map((row) => featureNames.map((name) => (row[name] - means[name]) / deviations[name]));
const distance = (a: number[], b: number[]) => a.reduce((sum, value, index) => sum + (value - b[index]) ** 2, 0);
const k = Math.max(1, Math.min(clusterCount, vectors.length));
const centers: number[][] = [vectors[0].slice()];
while (centers.length < k) {
  let bestIndex = 0;
  let bestDistance = -1;
  for (let index = 0; index < vectors.length; index++) {
    const nearest = Math.min(...centers.map((center) => distance(vectors[index], center)));
    if (nearest > bestDistance) {
      bestDistance = nearest;
      bestIndex = index;
    }
  }
  centers.push(vectors[bestIndex].slice());
}
let assignments = new Array(vectors.length).fill(0);
for (let iteration = 0; iteration < 30; iteration++) {
  const next = vectors.map((vector) => centers.reduce((best, center, index) =>
    distance(vector, center) < distance(vector, centers[best]) ? index : best, 0));
  const changed = next.some((value, index) => value !== assignments[index]);
  assignments = next;
  for (let cluster = 0; cluster < k; cluster++) {
    const members = vectors.filter((_, index) => assignments[index] === cluster);
    if (!members.length) continue;
    centers[cluster] = featureNames.map((_, feature) => members.reduce((sum, row) => sum + row[feature], 0) / members.length);
  }
  if (!changed && iteration > 0) break;
}

const average = (values: number[]) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const activeClusters = [...new Set(assignments)].sort((a, b) => a - b);
const summaries = activeClusters.map((cluster) => {
  const indexes = assignments.map((value, index) => value === cluster ? index : -1).filter((index) => index >= 0);
  const outcomes = indexes.map((index) => outcomeDelta(results[index]));
  const representative = indexes.reduce((best, index) =>
    distance(vectors[index], centers[cluster]) < distance(vectors[best], centers[cluster]) ? index : best, indexes[0]);
  const topSignals = featureNames
    .map((name, index) => ({ name, z: centers[cluster][index] }))
    .filter((item) => Math.abs(item.z) >= 0.35)
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
    .slice(0, 5);
  const kinds = new Map<string, number>();
  for (const index of indexes) {
    const row = results[index].disagreement;
    const name = `${row.guideType}→${row.expertType}`;
    kinds.set(name, (kinds.get(name) ?? 0) + 1);
  }
  return {
    cluster,
    size: indexes.length,
    improved: outcomes.filter((value) => value > 0).length,
    unchanged: outcomes.filter((value) => value === 0).length,
    worsened: outcomes.filter((value) => value < 0).length,
    averageOutcome: average(outcomes),
    kinds: Object.fromEntries([...kinds].sort((a, b) => b[1] - a[1])),
    topSignals,
    rawMeans: Object.fromEntries(featureNames.map((name) => [name, average(indexes.map((index) => raw[index][name]))])),
    representative,
  };
}).sort((a, b) => b.averageOutcome - a.averageOutcome || b.improved - a.improved);

console.log(`# 第${champion.generation}世代の概念プロトタイプ分析`);
const selectedLosses = results.filter((result) => !result.disagreement.guideWon).length;
console.log(`発見: ${games}局、重複除去後${ranked.length}局面、終局追跡${results.length}局面（敗戦${selectedLosses}・勝利${results.length - selectedLosses}）`);
console.log(`探索: 発見${depth}手、終局追跡${rolloutDepth}手、クラスタ数${summaries.length}、seed=${seed}`);
const allOutcomes = results.map(outcomeDelta);
console.log(`探索AI手による勝敗: 改善${allOutcomes.filter((value) => value > 0).length} / 同じ${allOutcomes.filter((value) => value === 0).length} / 悪化${allOutcomes.filter((value) => value < 0).length}`);

for (const [rank, summary] of summaries.entries()) {
  const result = results[summary.representative];
  const disagreement = result.disagreement;
  console.log(`\n## C${rank + 1}: ${summary.size}局面・改善${summary.improved}/同じ${summary.unchanged}/悪化${summary.worsened}`);
  console.log(`- 不一致型: ${Object.entries(summary.kinds).map(([name, count]) => `${name} ${count}`).join('、')}`);
  console.log(`- 特徴: ${summary.topSignals.map(({ name, z }) => `${featureLabels[name]} ${z > 0 ? '高' : '低'}(${z.toFixed(1)}σ)`).join('、') || '全体平均に近い'}`);
  console.log(`- 平均: ${summary.rawMeans.ply.toFixed(1)}手目、歩数リード${summary.rawMeans.raceLead.toFixed(2)}、壁差${summary.rawMeans.wallDiff.toFixed(2)}、即時差${summary.rawMeans.netGainDelta.toFixed(2)}、応手後歩数差${summary.rawMeans.replyRaceDelta.toFixed(2)}`);
  console.log(`- 代表: ${disagreement.position.ply}手目、${disagreement.guideMove}→${disagreement.expertMove}、regret ${disagreement.regret}、勝敗差 ${outcomeDelta(result) > 0 ? '改善' : outcomeDelta(result) < 0 ? '悪化' : '同じ'}`);
  console.log(`- 盤面: 駒 ${disagreement.position.pawns.join('/')}、横壁 [${disagreement.position.hWalls.join(' ')}]、縦壁 [${disagreement.position.vWalls.join(' ')}]`);
  console.log(`- 棋譜: ${disagreement.position.moves || '（初期局面）'}`);
}

console.log('\n## 読み方');
console.log('- クラスタは勝敗ラベルを使わず局面・候補手の特徴だけで作った。改善率の高い群でも、別seedと反例で条件を再確認する。');
console.log('- 次は上位クラスタの代表例と悪化例をLLMで比較し、計測可能な新特徴量候補へ変換する。');

writeFileSync('experiment-output.json', `${JSON.stringify({
  generation: champion.generation,
  source: { games, top, clusterCount: summaries.length, depth, rolloutDepth, seed },
  totals: {
    discovered: ranked.length,
    followed: results.length,
    improved: allOutcomes.filter((value) => value > 0).length,
    unchanged: allOutcomes.filter((value) => value === 0).length,
    worsened: allOutcomes.filter((value) => value < 0).length,
  },
  summaries: summaries.map((summary) => ({
    ...summary,
    representative: results[summary.representative],
  })),
}, null, 2)}\n`);
