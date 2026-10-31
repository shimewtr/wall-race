// チャンピオンと学習中重みで判断が違う局面を、両候補手から終局まで追跡する。
// 重みが動いた量ではなく、現在の学習方向が実際の勝敗を改善しているかを測る。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { transformState } from '../../src/analysis/symmetry';
import { FEATURE_NAMES, features, recentCells, search, withDefaults, type Feature, type Weights } from '../../src/engine/ai';
import {
  WALL_SLOTS,
  applyMove,
  initialState,
  isOver,
  moveToString,
  shortestDistance,
  type GameState,
  type Player,
} from '../../src/engine/game';
import type { Disagreement, DiscoveryPosition } from '../../src/guide/discover';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';
import { CounterfactualPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '80' },
    positions: { type: 'string', default: '600' },
    changes: { type: 'string', default: '100' },
    depth: { type: 'string', default: '3' },
    'rollout-depth': { type: 'string', default: '3' },
    'sample-every': { type: 'string', default: '4' },
    seed: { type: 'string', default: '410000000' },
  },
});

const root = join(import.meta.dirname, '..', '..');
const championFile = JSON.parse(readFileSync(join(root, 'training', 'champion.json'), 'utf8')) as { generation: number; weights: Partial<Weights> };
const stateFile = JSON.parse(readFileSync(join(root, 'training', 'state.json'), 'utf8')) as { theta: Partial<Weights>; totalIterations: number };
const champion = withDefaults(championFile.weights);
const training = withDefaults(stateFile.theta);
const games = Number(args.games);
const maxPositions = Number(args.positions);
const maxChanges = Number(args.changes);
const depth = Number(args.depth);
const rolloutDepth = Number(args['rollout-depth']);
const sampleEvery = Number(args['sample-every']);
const seed = Number(args.seed);

interface Sample { game: number; moves: string; state: GameState }
const samples: Sample[] = [];
for (let game = 0; game < games && samples.length < maxPositions; game++) {
  const rng = rngFrom(seed + game);
  const history = [initialState()];
  const moves: string[] = [];
  let state = history[0];
  while (!isOver(state) && samples.length < maxPositions) {
    if (state.ply % sampleEvery === game % sampleEvery) samples.push({ game: game + 1, moves: moves.join(' '), state });
    const move = search(state, {
      maxDepth: depthAt(state.ply, depth, OPENING_PLIES),
      weights: champion,
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
  for (let y = 0; y < WALL_SLOTS; y++) for (let x = 0; x < WALL_SLOTS; x++) {
    if (walls[y * WALL_SLOTS + x]) names.push(moveToString({ type: 'wall', x, y, o: orientation }));
  }
  return names;
};
const stateKey = (state: GameState) => {
  const encode = (value: GameState) => JSON.stringify([value.pawns, value.wallsLeft, Array.from(value.hWalls), Array.from(value.vWalls), value.turn]);
  const normal = encode(state);
  const mirrored = encode(transformState(state, 'left-right'));
  return normal < mirrored ? normal : mirrored;
};

interface Change {
  sample: Sample;
  disagreement: Disagreement;
  moveDeltas: Record<Feature, number>;
}
const changes: Change[] = [];
const seen = new Set<string>();
for (const sample of samples) {
  const base = search(sample.state, { maxDepth: depth, weights: champion, repeatPenalty: 0 });
  const candidate = search(sample.state, { maxDepth: depth, weights: training, repeatPenalty: 0 });
  if (moveToString(base.move) === moveToString(candidate.move)) continue;
  const key = stateKey(sample.state);
  if (seen.has(key)) continue;
  seen.add(key);
  const me = sample.state.turn;
  const op = (1 - me) as Player;
  const baseAfter = features(applyMove(sample.state, base.move));
  const candidateAfter = features(applyMove(sample.state, candidate.move));
  // applyMove後は相手手番なので、base - candidate が元の手番側から見た候補の改善量になる。
  const moveDeltas = Object.fromEntries(FEATURE_NAMES.map((name) => [name, baseAfter[name] - candidateAfter[name]])) as Record<Feature, number>;
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
    moveDeltas,
    disagreement: {
      position,
      guideMove: moveToString(base.move),
      expertMove: moveToString(candidate.move),
      guideType: base.move.type,
      expertType: candidate.move.type,
      guideScore: base.score,
      expertScore: candidate.score,
      regret: Math.abs(candidate.score - base.score),
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
const outcomes = await pool.runAll(changes.map(({ disagreement }) => ({ disagreement, weights: champion, depth: rolloutDepth })));
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
const kind = (index: number) => `${changes[index].disagreement.guideType}→${changes[index].disagreement.expertType}`;
const kinds = (source: number[]) => {
  const counts = new Map<string, number>();
  for (const index of source) counts.set(kind(index), (counts.get(kind(index)) ?? 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1]).map(([name, count]) => `${name} ${count}`).join('、') || 'なし';
};

console.log(`# 学習中重み対第${championFile.generation}世代の判断差`);
console.log(`学習時点: 累計${stateFile.totalIterations}反復、seed=${seed}`);
console.log(`収集: ${new Set(samples.map((sample) => sample.game)).size}局・${samples.length}局面、判断差${changes.length}`);
console.log(`終局追跡: 学習中重みの手で改善${improved.length} / 同じ${same.length} / 悪化${worsened.length}`);
console.log(`- 改善: ${kinds(improved)}`);
console.log(`- 悪化: ${kinds(worsened)}`);
console.log(`- 同じ: ${kinds(same)}`);

console.log('\n## 学習中候補−チャンピオン候補の手後特徴差');
const contrasts = FEATURE_NAMES.map((name) => ({
  name,
  improved: average(improved.map((index) => changes[index].moveDeltas[name])),
  worsened: average(worsened.map((index) => changes[index].moveDeltas[name])),
})).sort((a, b) => Math.abs(b.improved - b.worsened) - Math.abs(a.improved - a.worsened));
for (const row of contrasts) console.log(`- ${row.name}: 改善${row.improved.toFixed(2)} / 悪化${row.worsened.toFixed(2)} / 差${(row.improved - row.worsened).toFixed(2)}`);

console.log('\n## 重みの変化');
for (const name of FEATURE_NAMES) console.log(`- ${name}: ${champion[name].toFixed(3)} → ${training[name].toFixed(3)}（${(training[name] - champion[name]).toFixed(3)}）`);

const output = { generation: championFile.generation, totalIterations: stateFile.totalIterations, seed, samples: samples.length, changes: changes.length, improved: improved.length, same: same.length, worsened: worsened.length, contrasts };
writeFileSync('experiment-output.json', `${JSON.stringify(output, null, 2)}\n`);
