// 手引きと探索AIが異なる駒手を選ぶ局面を、元対局の勝敗を均衡させて終局追跡する。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { moveFromString } from '../../src/engine/game';
import type { CounterfactualResult } from '../../src/guide/counterfactual';
import type { Disagreement, PawnEffects } from '../../src/guide/discover';
import { CounterfactualPool, DiscoveryPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '480' },
    'per-outcome': { type: 'string', default: '60' },
    depth: { type: 'string', default: '3' },
    'rollout-depth': { type: 'string', default: '3' },
    seed: { type: 'string', default: '149000000' },
  },
});

const champion = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'));
const games = Number(args.games);
const perOutcome = Number(args['per-outcome']);
const depth = Number(args.depth);
const rolloutDepth = Number(args['rollout-depth']);
const seed = Number(args.seed);

const discoveryPool = new DiscoveryPool();
const found = await discoveryPool.runAll(
  Array.from({ length: games }, (_, i) => ({
    weights: champion.weights,
    depth,
    guidePlayer: (i % 2) as 0 | 1,
    seed: seed + i,
  })),
);
await discoveryPool.close();

const positionKey = (d: Disagreement) =>
  JSON.stringify([d.position.pawns, d.position.wallsLeft, d.position.hWalls, d.position.vWalls, d.position.guidePlayer]);
const unique = new Map<string, Disagreement>();
for (const disagreement of found.flatMap((game) => game.disagreements)) {
  if (!disagreement.pawnComparison || disagreement.regret < 1) continue;
  const key = `${disagreement.guideWon ? 'win' : 'loss'}:${positionKey(disagreement)}`;
  const previous = unique.get(key);
  if (!previous || disagreement.regret > previous.regret) unique.set(key, disagreement);
}

const choose = (won: boolean) => [...unique.values()]
  .filter((d) => d.guideWon === won)
  .sort((a, b) => b.regret - a.regret)
  .slice(0, perOutcome);
const selected = [...choose(false), ...choose(true)];

const counterfactualPool = new CounterfactualPool();
const results = await counterfactualPool.runAll(
  selected.map((disagreement) => ({ disagreement, weights: champion.weights, depth: rolloutDepth })),
);
await counterfactualPool.close();

const score = (result: CounterfactualResult, branch: 'guide' | 'expert') => {
  const winner = result[branch].winner;
  return winner === null ? 0.5 : winner === result.disagreement.position.guidePlayer ? 1 : 0;
};
const effect = (result: CounterfactualResult, branch: 'guide' | 'expert'): PawnEffects =>
  result.disagreement.pawnComparison![branch];
const average = (values: number[]) => values.length
  ? values.reduce((sum, value) => sum + value, 0) / values.length
  : 0;
const pct = (value: number, total: number) => `${total ? (value / total * 100).toFixed(1) : '0.0'}%`;
const movement = (value: number) => value > 0 ? '前進' : value < 0 ? '遠回り' : '同距離';
const replyType = (result: CounterfactualResult, branch: 'guide' | 'expert') =>
  moveFromString(result[branch].moves[1] ?? '')?.type ?? '終局';

interface Summary {
  name: string;
  count: number;
  improved: number;
  unchanged: number;
  worsened: number;
  deltas: Record<string, number>;
  expertReplyWalls: number;
  guideReplyWalls: number;
}

function summarize(name: string, group: CounterfactualResult[]): Summary {
  const improved = group.filter((result) => score(result, 'expert') > score(result, 'guide')).length;
  const worsened = group.filter((result) => score(result, 'expert') < score(result, 'guide')).length;
  const delta = (pick: (value: PawnEffects) => number) =>
    average(group.map((result) => pick(effect(result, 'expert')) - pick(effect(result, 'guide'))));
  return {
    name,
    count: group.length,
    improved,
    unchanged: group.length - improved - worsened,
    worsened,
    deltas: {
      progress: delta((value) => value.progress),
      centerOffset: delta((value) => value.centerOffset),
      threatBalance: delta((value) => value.myThreat - value.opThreat),
      routeBalance: delta((value) => value.myRoutes - value.opRoutes),
      adjacent: delta((value) => Number(value.adjacent)),
      backtrack: delta((value) => Number(value.backtrack)),
    },
    expertReplyWalls: group.filter((result) => replyType(result, 'expert') === 'wall').length,
    guideReplyWalls: group.filter((result) => replyType(result, 'guide') === 'wall').length,
  };
}

const originalLosses = results.filter((result) => !result.disagreement.guideWon);
const originalWins = results.filter((result) => result.disagreement.guideWon);
const improved = results.filter((result) => score(result, 'expert') > score(result, 'guide'));
const nonImproved = results.filter((result) => score(result, 'expert') <= score(result, 'guide'));
const summaries = [
  summarize('元対局で敗戦', originalLosses),
  summarize('元対局で勝利', originalWins),
  summarize('探索AI手で勝敗改善', improved),
  summarize('勝敗非改善', nonImproved),
];

console.log(`# 駒手→駒手の反実仮想分析: 手引き AI 対 第 ${champion.generation} 世代`);
console.log(`発見${games}局、選択${results.length}局面（元対局の敗戦${originalLosses.length} / 勝利${originalWins.length}）、探索${depth}手・終局追跡${rolloutDepth}手`);
console.log('各局面で最初の駒手だけを変え、その後は同じ探索AI同士で終局まで指しています。');

console.log('\n## 勝敗と特徴差');
for (const row of summaries) {
  console.log(`- ${row.name}（${row.count}）: 改善${row.improved} / 同じ${row.unchanged} / 悪化${row.worsened}`);
  console.log(
    `  - 探索AI−手引き: 前進${row.deltas.progress.toFixed(2)}、中央からの距離${row.deltas.centerOffset.toFixed(2)}、壁脅威差${row.deltas.threatBalance.toFixed(2)}、経路数差${row.deltas.routeBalance.toFixed(2)}、隣接${row.deltas.adjacent.toFixed(2)}、戻り${row.deltas.backtrack.toFixed(2)}`,
  );
  console.log(`  - 相手が次に壁: 探索AI手後${pct(row.expertReplyWalls, row.count)} / 手引き手後${pct(row.guideReplyWalls, row.count)}`);
}

console.log('\n## 前進量の組み合わせ');
const movementCounts = new Map<string, number>();
for (const result of results) {
  const key = `${movement(effect(result, 'guide').progress)}→${movement(effect(result, 'expert').progress)}`;
  movementCounts.set(key, (movementCounts.get(key) ?? 0) + 1);
}
for (const [name, count] of [...movementCounts].sort((a, b) => b[1] - a[1])) console.log(`- ${name}: ${count}`);

console.log('\n## 勝敗が変わった上位例');
const changed = results
  .filter((result) => score(result, 'guide') !== score(result, 'expert'))
  .sort((a, b) => b.disagreement.regret - a.disagreement.regret);
for (const [index, result] of changed.slice(0, 12).entries()) {
  const d = result.disagreement;
  const guide = effect(result, 'guide');
  const expert = effect(result, 'expert');
  console.log(
    `- ${index + 1}. ${d.guideWon ? '元勝利' : '元敗戦'}・${d.position.ply}手目・${d.guideMove}→${d.expertMove}・regret ${d.regret}: ${score(result, 'guide')}→${score(result, 'expert')}、前進 ${guide.progress}→${expert.progress}、中央 ${guide.centerOffset}→${expert.centerOffset}、相手応手 ${replyType(result, 'guide')}→${replyType(result, 'expert')}`,
  );
}

writeFileSync('experiment-output.json', `${JSON.stringify({
  source: { generation: champion.generation, games, perOutcome, depth, rolloutDepth, seed },
  summaries,
  movementCounts: Object.fromEntries(movementCounts),
  changed: changed.map((result) => ({
    disagreement: result.disagreement,
    guide: result.guide,
    expert: result.expert,
  })),
}, null, 2)}\n`);
