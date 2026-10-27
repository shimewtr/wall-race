// 3手読みと4手読みで判断が違う局面を、両候補手から終局まで指し継いで比較する。
// 深い読みへの一致率ではなく、実際に勝敗を改善する判断差だけを新特徴量探索へ渡す。
// 重い処理なのでGitHub ActionsのExperimentで実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { transformState } from '../../src/analysis/symmetry';
import { features as evaluateFeatures, recentCells, search, type Feature, type Weights } from '../../src/engine/ai';
import {
  type GameState,
  type Move,
  type Player,
  WALL_SLOTS,
  applyMove,
  initialState,
  isOver,
  moveFromString,
  moveToString,
  shortestDistance,
} from '../../src/engine/game';
import type { Disagreement, DiscoveryPosition } from '../../src/guide/discover';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';
import { CounterfactualPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '40' },
    positions: { type: 'string', default: '240' },
    changes: { type: 'string', default: '60' },
    'sample-every': { type: 'string', default: '4' },
    'rollout-depth': { type: 'string', default: '3' },
    seed: { type: 'string', default: '112000000' },
  },
});

const root = join(import.meta.dirname, '..', '..');
const champion = JSON.parse(readFileSync(join(root, 'training', 'champion.json'), 'utf8')) as {
  generation: number;
  weights: Weights;
};
const games = Number(args.games);
const maxPositions = Number(args.positions);
const maxChanges = Number(args.changes);
const sampleEvery = Number(args['sample-every']);
const rolloutDepth = Number(args['rollout-depth']);
const seed = Number(args.seed);

interface Sample {
  game: number;
  moves: string;
  state: GameState;
}

const samples: Sample[] = [];
for (let game = 0; game < games && samples.length < maxPositions; game++) {
  const rng = rngFrom(seed + game);
  const history: GameState[] = [initialState()];
  const moves: string[] = [];
  let state = history[0];
  while (!isOver(state) && samples.length < maxPositions) {
    if (state.ply % sampleEvery === game % sampleEvery) {
      samples.push({ game: game + 1, moves: moves.join(' '), state });
    }
    const move = search(state, {
      maxDepth: depthAt(state.ply, 3, OPENING_PLIES),
      weights: champion.weights,
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

const wallNames = (state: GameState, orientation: 'h' | 'v') => {
  const walls = orientation === 'h' ? state.hWalls : state.vWalls;
  const names: string[] = [];
  for (let y = 0; y < WALL_SLOTS; y++) {
    for (let x = 0; x < WALL_SLOTS; x++) {
      if (walls[y * WALL_SLOTS + x]) names.push(moveToString({ type: 'wall', x, y, o: orientation }));
    }
  }
  return names;
};
const sameMove = (a: Move, b: Move) => moveToString(a) === moveToString(b);
const phase = (ply: number) => ply < 20 ? '序盤' : ply < 40 ? '中盤' : '終盤';
const stateKey = (state: GameState) => {
  const encode = (value: GameState) => JSON.stringify([
    value.pawns,
    value.wallsLeft,
    Array.from(value.hWalls),
    Array.from(value.vWalls),
    value.turn,
  ]);
  const normal = encode(state);
  const mirrored = encode(transformState(state, 'left-right'));
  return normal < mirrored ? normal : mirrored;
};

interface Change {
  sample: Sample;
  disagreement: Disagreement;
  shallowRootGap: number;
  values: Record<Feature, number>;
  moveDeltas: Record<Feature | 'raceLeadAfter' | 'wallDiffAfter', number>;
}
const changes: Change[] = [];
const seen = new Set<string>();
for (const sample of samples) {
  const depth3 = search(sample.state, { maxDepth: 3, weights: champion.weights, repeatPenalty: 0 });
  const depth4 = search(sample.state, { maxDepth: 4, weights: champion.weights, repeatPenalty: 0 });
  if (sameMove(depth3.move, depth4.move)) continue;
  const key = stateKey(sample.state);
  if (seen.has(key)) continue;
  seen.add(key);
  const me = sample.state.turn;
  const op = (1 - me) as Player;
  const shallowAfter = applyMove(sample.state, depth3.move);
  const deepAfter = applyMove(sample.state, depth4.move);
  // applyMove後は相手手番なので、featuresの符号を反転すると元の手番側から見た値になる。
  const shallowFeatures = evaluateFeatures(shallowAfter);
  const deepFeatures = evaluateFeatures(deepAfter);
  const moveDeltas = Object.fromEntries(Object.keys(shallowFeatures).map((feature) => [
    feature,
    shallowFeatures[feature as Feature] - deepFeatures[feature as Feature],
  ])) as Change['moveDeltas'];
  moveDeltas.raceLeadAfter = (
    shortestDistance(deepAfter, op) - shortestDistance(deepAfter, me)
  ) - (
    shortestDistance(shallowAfter, op) - shortestDistance(shallowAfter, me)
  );
  moveDeltas.wallDiffAfter = (
    deepAfter.wallsLeft[me] - deepAfter.wallsLeft[op]
  ) - (
    shallowAfter.wallsLeft[me] - shallowAfter.wallsLeft[op]
  );
  const position: DiscoveryPosition = {
    ply: sample.state.ply,
    guidePlayer: me,
    pawns: sample.state.pawns.map((pawn) => moveToString({ type: 'pawn', ...pawn })) as [string, string],
    wallsLeft: [...sample.state.wallsLeft],
    hWalls: wallNames(sample.state, 'h'),
    vWalls: wallNames(sample.state, 'v'),
    moves: sample.moves,
  };
  changes.push({
    sample,
    shallowRootGap: depth3.rootGap ?? Infinity,
    values: evaluateFeatures(sample.state),
    moveDeltas,
    disagreement: {
      position,
      guideMove: moveToString(depth3.move),
      expertMove: moveToString(depth4.move),
      guideType: depth3.move.type,
      expertType: depth4.move.type,
      guideScore: depth3.score,
      expertScore: depth4.score,
      regret: Math.abs(depth4.score - depth3.score),
      myDist: shortestDistance(sample.state, me),
      opDist: shortestDistance(sample.state, op),
      myWalls: sample.state.wallsLeft[me],
      opWalls: sample.state.wallsLeft[op],
      myThreat: 0,
      opThreat: 0,
      myRoutes: 0,
      opRoutes: 0,
    },
  });
  if (changes.length >= maxChanges) break;
}

const pool = new CounterfactualPool();
const outcomes = await pool.runAll(changes.map(({ disagreement }) => ({
  disagreement,
  weights: champion.weights,
  depth: rolloutDepth,
})));
await pool.close();

const score = (winner: Player | null, player: Player) => winner === null ? 0.5 : winner === player ? 1 : 0;
const deltaAt = (index: number) => {
  const result = outcomes[index];
  const player = result.disagreement.position.guidePlayer;
  return score(result.expert.winner, player) - score(result.guide.winner, player);
};
const indexes = changes.map((_, index) => index);
const improved = indexes.filter((index) => deltaAt(index) > 0);
const worsened = indexes.filter((index) => deltaAt(index) < 0);
const same = indexes.filter((index) => deltaAt(index) === 0);
const average = (values: number[]) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const fields = ['ply', 'raceLead', 'wallDiff', 'pathDiff', 'rowDiff', 'wallThreat', 'netWallThreat', 'wallChain', 'routeDiversity'] as const;
const valueAt = (index: number, field: typeof fields[number]) => {
  const change = changes[index];
  if (field === 'ply') return change.sample.state.ply;
  if (field === 'raceLead') return change.disagreement.opDist - change.disagreement.myDist;
  if (field === 'wallDiff') return change.disagreement.myWalls - change.disagreement.opWalls;
  return change.values[field];
};
const contrasts = fields.map((field) => ({
  field,
  improved: average(improved.map((index) => valueAt(index, field))),
  worsened: average(worsened.map((index) => valueAt(index, field))),
})).sort((a, b) => Math.abs(b.improved - b.worsened) - Math.abs(a.improved - a.worsened));
const moveFields = [...Object.keys(changes[0]?.moveDeltas ?? {})] as Array<keyof Change['moveDeltas']>;
const moveContrasts = moveFields.map((field) => ({
  field,
  improved: average(improved.map((index) => changes[index].moveDeltas[field])),
  worsened: average(worsened.map((index) => changes[index].moveDeltas[field])),
})).sort((a, b) => Math.abs(b.improved - b.worsened) - Math.abs(a.improved - a.worsened));
const gapGroups = [
  ['改善', improved],
  ['同じ', same],
  ['悪化', worsened],
] as const;

const kind = (index: number) => `${changes[index].disagreement.guideType}→${changes[index].disagreement.expertType}`;
const group = (name: string, candidates: number[]) => {
  const counts = new Map<string, number>();
  for (const index of candidates) {
    const key = `${phase(changes[index].sample.state.ply)} ${kind(index)}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  console.log(`- ${name}: ${[...counts].sort((a, b) => b[1] - a[1]).map(([key, count]) => `${key} ${count}`).join('、') || 'なし'}`);
};

console.log(`# 第${champion.generation}世代の3手→4手反実仮想`);
console.log(`収集: ${new Set(samples.map((sample) => sample.game)).size}局・${samples.length}局面、判断差${changes.length}、seed=${seed}`);
console.log(`終局追跡: 4手側が改善${improved.length} / 同じ${same.length} / 悪化${worsened.length}（追跡${rolloutDepth}手読み）`);
group('改善', improved);
group('悪化', worsened);
group('同じ', same);

const moveTypes = (moves: string[]) => moves.slice(0, 4).map((text) => {
  const move = moveFromString(text);
  return move?.type === 'wall' ? 'W' : move?.type === 'pawn' ? 'P' : '?';
}).join('');
console.log('\n## 壁→前進の直後4手');
for (const [name, candidates] of [['改善', improved], ['同じ', same], ['悪化', worsened]] as const) {
  const patterns = new Map<string, number>();
  for (const index of candidates.filter((value) => kind(value) === 'wall→pawn')) {
    const key = `3手側 ${moveTypes(outcomes[index].guide.moves)} / 4手側 ${moveTypes(outcomes[index].expert.moves)}`;
    patterns.set(key, (patterns.get(key) ?? 0) + 1);
  }
  console.log(`- ${name}: ${[...patterns].sort((a, b) => b[1] - a[1]).map(([key, count]) => `${key} ${count}`).join('、') || 'なし'}`);
}
console.log('\n## 改善例と悪化例の平均差');
for (const row of contrasts) console.log(`- ${row.field}: 改善${row.improved.toFixed(2)} / 悪化${row.worsened.toFixed(2)} / 差${(row.improved - row.worsened).toFixed(2)}`);
console.log('\n## 4手候補−3手候補の手後特徴差');
for (const row of moveContrasts) console.log(`- ${row.field}: 改善${row.improved.toFixed(2)} / 悪化${row.worsened.toFixed(2)} / 差${(row.improved - row.worsened).toFixed(2)}`);
console.log('\n## 3手読みの1位・2位評価差');
for (const [name, candidates] of gapGroups) {
  const gaps = candidates.map((index) => changes[index].shallowRootGap).sort((a, b) => a - b);
  const median = gaps.length ? gaps[Math.floor(gaps.length / 2)] : 0;
  console.log(`- ${name}: 平均${average(gaps).toFixed(2)} / 中央${median.toFixed(2)} / 5点以内${gaps.filter((gap) => gap <= 5).length}/${gaps.length} / 10点以内${gaps.filter((gap) => gap <= 10).length}/${gaps.length}`);
}
console.log('\n## 代表例');
for (const [name, candidates] of [['改善', improved], ['悪化', worsened]] as const) {
  for (const index of candidates.slice(0, 8)) {
    const row = changes[index];
    console.log(`- ${name}・対局${row.sample.game}・${row.sample.state.ply}手目: ${row.disagreement.guideMove}→${row.disagreement.expertMove}（${kind(index)}）、残り歩数${row.disagreement.myDist}/${row.disagreement.opDist}、壁${row.disagreement.myWalls}/${row.disagreement.opWalls}`);
    console.log(`  棋譜: ${row.sample.moves || '（初期局面）'}`);
  }
}

writeFileSync('experiment-output.json', `${JSON.stringify({
  generation: champion.generation,
  source: { games, maxPositions, maxChanges, sampleEvery, rolloutDepth, seed },
  totals: { sampled: samples.length, changes: changes.length, improved: improved.length, same: same.length, worsened: worsened.length },
  contrasts,
  moveContrasts,
  examples: {
    improved: improved.slice(0, 8).map((index) => ({ change: changes[index].disagreement, outcome: outcomes[index] })),
    worsened: worsened.slice(0, 8).map((index) => ({ change: changes[index].disagreement, outcome: outcomes[index] })),
  },
  rows: indexes.map((index) => ({
    delta: deltaAt(index),
    shallowRootGap: changes[index].shallowRootGap,
    values: changes[index].values,
    moveDeltas: changes[index].moveDeltas,
    change: changes[index].disagreement,
    outcome: outcomes[index],
  })),
}, null, 2)}\n`);
