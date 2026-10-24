// 既存特徴量を1つずつ無効化し、探索 AI の判断がどこで変わるかを比較する。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/ablation.ts --games 20 --positions 120 --depth 3 --sample-every 4

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { FEATURE_NAMES, recentCells, search, type Feature, type Weights } from '../../src/engine/ai';
import { type GameState, type Move, applyMove, initialState, isOver, moveToString } from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '20' },
    positions: { type: 'string', default: '120' },
    depth: { type: 'string', default: '3' },
    'sample-every': { type: 'string', default: '4' },
    features: { type: 'string', default: 'all' },
  },
});

const champion = JSON.parse(
  readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'),
);
const weights = champion.weights as Weights;
const games = Number(args.games);
const maxPositions = Number(args.positions);
const depth = Number(args.depth);
const sampleEvery = Number(args['sample-every']);
const requested = args.features === 'all'
  ? FEATURE_NAMES
  : args.features.split(',').map((name) => name.trim() as Feature).filter(Boolean);
for (const feature of requested) {
  if (!FEATURE_NAMES.includes(feature)) throw new Error(`unknown feature: ${feature}`);
}

interface Choice {
  move: string;
  type: Move['type'];
  score: number;
}

interface Position {
  game: number;
  ply: number;
  position: string;
  baseline: Choice;
  depthFour: Choice;
  ablated: Map<Feature, Choice>;
}

const choice = (result: ReturnType<typeof search>): Choice => ({
  move: moveToString(result.move),
  type: result.move.type,
  score: result.score,
});
const positions: Position[] = [];

for (let game = 0; game < games && positions.length < maxPositions; game++) {
  const rng = rngFrom(75_000_000 + game);
  const history: GameState[] = [initialState()];
  const moves: string[] = [];
  let state = history[0];

  while (!isOver(state) && positions.length < maxPositions) {
    if (state.ply % sampleEvery === game % sampleEvery) {
      const baseline = choice(search(state, { maxDepth: depth, weights, repeatPenalty: 0 }));
      const depthFour = choice(search(state, { maxDepth: depth + 1, weights, repeatPenalty: 0 }));
      const ablated = new Map<Feature, Choice>();
      for (const feature of requested) {
        ablated.set(
          feature,
          choice(search(state, { maxDepth: depth, weights: { ...weights, [feature]: 0 }, repeatPenalty: 0 })),
        );
      }
      positions.push({ game: game + 1, ply: state.ply, position: moves.join(' '), baseline, depthFour, ablated });
    }

    const move = search(state, {
      maxDepth: depthAt(state.ply, depth, OPENING_PLIES),
      weights,
      margin: 1,
      pickFromTop: pickAt(state.ply, OPENING_PLIES),
      rng,
      recent: recentCells(history, state.turn),
    }).move;
    moves.push(moveToString(move));
    state = applyMove(state, move);
    history.push(state);
  }
}

const pct = (count: number, total: number) =>
  total === 0 ? '-' : `${((count / total) * 100).toFixed(1)}%`;
const average = (values: number[]) =>
  values.length === 0 ? '-' : (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2);
const horizonSensitive = new Set(
  positions.map((row, index) => row.baseline.move !== row.depthFour.move ? index : -1).filter((index) => index >= 0),
);

interface FeatureResult {
  feature: Feature;
  changed: Set<number>;
  typeChanges: number;
  sensitiveChanges: number;
  stableChanges: number;
  averageScoreDelta: string;
}

const results: FeatureResult[] = requested.map((feature) => {
  const changed = new Set<number>();
  let typeChanges = 0;
  let sensitiveChanges = 0;
  let stableChanges = 0;
  const scoreDeltas: number[] = [];
  for (const [index, row] of positions.entries()) {
    const ablated = row.ablated.get(feature)!;
    scoreDeltas.push(Math.abs(ablated.score - row.baseline.score));
    if (ablated.move === row.baseline.move) continue;
    changed.add(index);
    if (ablated.type !== row.baseline.type) typeChanges++;
    if (horizonSensitive.has(index)) sensitiveChanges++;
    else stableChanges++;
  }
  return {
    feature,
    changed,
    typeChanges,
    sensitiveChanges,
    stableChanges,
    averageScoreDelta: average(scoreDeltas),
  };
}).sort((a, b) => b.changed.size - a.changed.size);

console.log(`# 第 ${champion.generation} 世代の特徴量アブレーション`);
console.log();
console.log(`収集: ${new Set(positions.map((row) => row.game)).size} 局から ${positions.length} 局面`);
console.log(`探索: 基準${depth}手読み、深さ依存判定${depth + 1}手読み、${sampleEvery}手間隔`);
console.log(`深さ依存局面: ${horizonSensitive.size} / ${positions.length}（${pct(horizonSensitive.size, positions.length)}）`);
console.log('変更率は「その特徴だけ重み0」にしたとき、基準AIと別の最善手を選んだ割合です。');

console.log('\n## 特徴量別の判断への影響');
console.log('| 特徴量 | 重み | 最善手変更 | 種類変更 | 深さ依存局面 | 安定局面 | 平均評価差 |');
console.log('| --- | ---: | ---: | ---: | ---: | ---: | ---: |');
for (const result of results) {
  console.log(
    `| ${result.feature} | ${weights[result.feature].toFixed(3)} | ${result.changed.size} (${pct(result.changed.size, positions.length)}) | ${result.typeChanges} | ${result.sensitiveChanges}/${horizonSensitive.size} | ${result.stableChanges}/${positions.length - horizonSensitive.size} | ${result.averageScoreDelta} |`,
  );
}

interface Overlap {
  a: Feature;
  b: Feature;
  intersection: number;
  union: number;
  jaccard: number;
}
const overlaps: Overlap[] = [];
for (let i = 0; i < results.length; i++) {
  for (let j = i + 1; j < results.length; j++) {
    const a = results[i];
    const b = results[j];
    const intersection = [...a.changed].filter((index) => b.changed.has(index)).length;
    const union = new Set([...a.changed, ...b.changed]).size;
    if (intersection === 0 || union === 0) continue;
    overlaps.push({ a: a.feature, b: b.feature, intersection, union, jaccard: intersection / union });
  }
}
overlaps.sort((a, b) => b.jaccard - a.jaccard || b.intersection - a.intersection);
console.log('\n## 判断変更局面の重なり');
if (overlaps.length === 0) console.log('- 重なりなし');
for (const overlap of overlaps.slice(0, 12)) {
  console.log(`- ${overlap.a} × ${overlap.b}: 共通${overlap.intersection} / 和集合${overlap.union}（Jaccard ${overlap.jaccard.toFixed(2)}）`);
}

console.log('\n## 読み方');
console.log('- 変更率が高い特徴量ほど現在の方策を左右するが、強さへの貢献を意味するとは限らない。');
console.log('- 深さ依存局面に偏る特徴量は、浅い探索を補うか、探索と二重評価している可能性がある。');
console.log('- 重なりが大きい組は冗長候補。上位だけを400局比較し、実際の勝率で採否を決める。');
