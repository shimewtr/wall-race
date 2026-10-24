// 手引き AI と探索 AI の不一致について、両方の候補手から探索 AI 同士で終局まで指し継ぐ。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/counterfactual.ts --games 120 --top 30 --depth 3 --rollout-depth 3

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import type { CounterfactualResult } from '../../src/guide/counterfactual';
import type { Disagreement } from '../../src/guide/discover';
import { CounterfactualPool, DiscoveryPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '120' },
    top: { type: 'string', default: '30' },
    depth: { type: 'string', default: '3' },
    'rollout-depth': { type: 'string', default: '3' },
  },
});

const champion = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'));
const games = Number(args.games);
const top = Number(args.top);
const depth = Number(args.depth);
const rolloutDepth = Number(args['rollout-depth']);

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

console.log(`# 反実仮想分析: 手引き AI 対 第 ${champion.generation} 世代`);
console.log();
console.log(`発見対局: ${games} 局（${depth} 手読み）`);
console.log(`終局追跡: 敗戦中・推定損失1点以上の上位 ${results.length} 局面（${rolloutDepth} 手読み）`);
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

console.log('\n## 勝敗が変わった局面');
const changed = results.filter((result) => score(result, 'guide') !== score(result, 'expert'));
if (changed.length === 0) console.log('- なし');
for (const [i, result] of changed.entries()) {
  const d = result.disagreement;
  console.log(`\n### ${i + 1}. ${d.position.ply} 手目・${kind(result)}・regret ${d.regret}`);
  console.log(`- 手引き ${d.guideMove}: ${resultName(result, 'guide')}（残り ${result.guide.remainingPlies} 手）`);
  console.log(`- 探索AI ${d.expertMove}: ${resultName(result, 'expert')}（残り ${result.expert.remainingPlies} 手）`);
  console.log(`- 残り歩数: 自分 ${d.myDist} / 相手 ${d.opDist}、壁: 自分 ${d.myWalls} / 相手 ${d.opWalls}`);
  console.log(`- 盤面: 駒 ${d.position.pawns.join(' / ')}、横壁 [${d.position.hWalls.join(' ')}]、縦壁 [${d.position.vWalls.join(' ')}]`);
  console.log(`- 直前までの棋譜: ${d.position.moves || '（初期局面）'}`);
}
