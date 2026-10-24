// 同じ局面を複数の深さで探索し、読みが深くなると判断がどう変わるかを集計する。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/horizon.ts --games 20 --positions 120 --max-depth 4 --sample-every 4

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { recentCells, search } from '../../src/engine/ai';
import { type GameState, type Move, type Player, applyMove, initialState, isOver, moveToString, shortestDistance } from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '20' },
    positions: { type: 'string', default: '120' },
    'max-depth': { type: 'string', default: '4' },
    'sample-every': { type: 'string', default: '4' },
    examples: { type: 'string', default: '12' },
  },
});

const champion = JSON.parse(
  readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'),
);
const games = Number(args.games);
const maxPositions = Number(args.positions);
const maxDepth = Number(args['max-depth']);
const sampleEvery = Number(args['sample-every']);
const maxExamples = Number(args.examples);
if (maxDepth < 2) throw new Error('max-depth must be at least 2');

interface DepthChoice {
  depth: number;
  move: string;
  type: Move['type'];
  orientation: 'h' | 'v' | null;
  score: number;
  nodes: number;
}

interface Observation {
  game: number;
  ply: number;
  turn: number;
  position: string;
  myDist: number;
  opDist: number;
  myWalls: number;
  opWalls: number;
  choices: DepthChoice[];
}

const observations: Observation[] = [];

for (let game = 0; game < games && observations.length < maxPositions; game++) {
  const rng = rngFrom(73_000_000 + game);
  const history: GameState[] = [initialState()];
  const moves: string[] = [];
  let state = history[0];

  while (!isOver(state) && observations.length < maxPositions) {
    if (state.ply % sampleEvery === game % sampleEvery) {
      const choices: DepthChoice[] = [];
      for (let depth = 1; depth <= maxDepth; depth++) {
        const result = search(state, {
          maxDepth: depth,
          weights: champion.weights,
          repeatPenalty: 0,
        });
        choices.push({
          depth,
          move: moveToString(result.move),
          type: result.move.type,
          orientation: result.move.type === 'wall' ? result.move.o : null,
          score: result.score,
          nodes: result.nodes,
        });
      }
      observations.push({
        game: game + 1,
        ply: state.ply,
        turn: state.turn,
        position: moves.join(' '),
        myDist: shortestDistance(state, state.turn),
        opDist: shortestDistance(state, (1 - state.turn) as Player),
        myWalls: state.wallsLeft[state.turn],
        opWalls: state.wallsLeft[(1 - state.turn) as Player],
        choices,
      });
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

const pct = (count: number, total: number) =>
  total === 0 ? '-' : `${((count / total) * 100).toFixed(1)}%`;
const average = (values: number[]) =>
  values.length === 0 ? '-' : (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(1);
const sign = (score: number) => (score > 1e-9 ? 1 : score < -1e-9 ? -1 : 0);
const transitionName = (before: DepthChoice, after: DepthChoice) => {
  if (before.move === after.move) return '同じ手';
  if (before.type !== after.type) return `${before.type === 'pawn' ? '前進' : '壁'}→${after.type === 'pawn' ? '前進' : '壁'}`;
  if (before.type === 'wall' && before.orientation !== after.orientation) return '壁の向き変更';
  return before.type === 'pawn' ? '別の移動先' : '別の壁';
};

console.log(`# 第 ${champion.generation} 世代の読み深さ分析`);
console.log();
console.log(`収集: ${new Set(observations.map((row) => row.game)).size} 局から ${observations.length} 局面`);
console.log(`探索: 1〜${maxDepth}手読み、${sampleEvery}手間隔、履歴ペナルティなし`);

console.log('\n## 深さを1段増やしたときの判断変化');
for (let depth = 1; depth < maxDepth; depth++) {
  const pairs = observations.map((row) => [row.choices[depth - 1], row.choices[depth]] as const);
  const moveChanges = pairs.filter(([before, after]) => before.move !== after.move);
  const typeChanges = moveChanges.filter(([before, after]) => before.type !== after.type);
  const signChanges = pairs.filter(([before, after]) => sign(before.score) !== sign(after.score));
  console.log(`\n### ${depth}手 → ${depth + 1}手`);
  console.log(`- 最善手変更: ${moveChanges.length} / ${pairs.length}（${pct(moveChanges.length, pairs.length)}）`);
  console.log(`- 前進・壁の変更: ${typeChanges.length} / ${pairs.length}（${pct(typeChanges.length, pairs.length)}）`);
  console.log(`- 生の評価値の符号反転: ${signChanges.length} / ${pairs.length}（${pct(signChanges.length, pairs.length)}）`);
  const kinds = new Map<string, number>();
  for (const pair of moveChanges) {
    const name = transitionName(...pair);
    kinds.set(name, (kinds.get(name) ?? 0) + 1);
  }
  console.log(`- 内訳: ${[...kinds].sort((a, b) => b[1] - a[1]).map(([name, count]) => `${name} ${count}`).join(' / ') || '変更なし'}`);
}

console.log('\n## 判断が安定する深さ');
for (let depth = 1; depth <= maxDepth; depth++) {
  const stable = observations.filter((row) => {
    const move = row.choices[depth - 1].move;
    return row.choices.slice(depth - 1).every((choice) => choice.move === move);
  }).length;
  console.log(`- ${depth}手読み以降で同じ手: ${stable} / ${observations.length}（${pct(stable, observations.length)}）`);
}
const oscillations = observations.filter((row) =>
  row.choices.some((choice, index, choices) => index >= 2 && choice.move === choices[index - 2].move && choice.move !== choices[index - 1].move),
);
console.log(`- 一度変えて元の手へ戻る局面: ${oscillations.length} / ${observations.length}（${pct(oscillations.length, observations.length)}）`);
const oddEvenOscillations = observations.filter(
  (row) =>
    maxDepth >= 4 &&
    row.choices[0].move === row.choices[2].move &&
    row.choices[1].move === row.choices[3].move &&
    row.choices[0].move !== row.choices[1].move,
);
if (maxDepth >= 4) {
  console.log(`- 1=3手・2=4手の偶奇振動: ${oddEvenOscillations.length} / ${observations.length}（${pct(oddEvenOscillations.length, observations.length)}）`);
}
console.log('- 注: 生の評価値は探索の深さごとに到達局面が違うため、深さをまたいだ符号だけを勝敗予測として使わず、選択手の変化と合わせて読む。');

const lastDepthChanges = observations.filter(
  (row) => row.choices[maxDepth - 2].move !== row.choices[maxDepth - 1].move,
);
console.log(`\n## ${maxDepth - 1}手→${maxDepth}手で変わる局面のフェーズ`);
for (const [name, includes] of [
  ['序盤（0〜19手）', (row: Observation) => row.ply < 20],
  ['中盤（20〜39手）', (row: Observation) => row.ply >= 20 && row.ply < 40],
  ['終盤（40手以降）', (row: Observation) => row.ply >= 40],
] as const) {
  const all = observations.filter(includes);
  const changed = lastDepthChanges.filter(includes);
  console.log(`- ${name}: ${changed.length} / ${all.length}（${pct(changed.length, all.length)}）`);
}
console.log(
  `- 変更局面の平均: 自分あと${average(lastDepthChanges.map((row) => row.myDist))}歩 / 相手あと${average(lastDepthChanges.map((row) => row.opDist))}歩 / 自分の壁${average(lastDepthChanges.map((row) => row.myWalls))}枚 / 相手の壁${average(lastDepthChanges.map((row) => row.opWalls))}枚`,
);

console.log('\n## 深さ別の探索量');
for (let depth = 1; depth <= maxDepth; depth++) {
  console.log(`- ${depth}手読み: 平均 ${average(observations.map((row) => row.choices[depth - 1].nodes))} ノード`);
}

const lastChanges = lastDepthChanges
  .sort((a, b) => {
    const ac = a.choices[maxDepth - 2].type !== a.choices[maxDepth - 1].type ? 1 : 0;
    const bc = b.choices[maxDepth - 2].type !== b.choices[maxDepth - 1].type ? 1 : 0;
    return bc - ac;
  })
  .slice(0, maxExamples);
console.log(`\n## ${maxDepth - 1}手と${maxDepth}手で判断が違う例`);
if (lastChanges.length === 0) console.log('- なし');
for (const row of lastChanges) {
  console.log(`\n### 対局${row.game}・${row.ply}手目・手番${row.turn}`);
  console.log(`- ${row.choices.map((choice) => `${choice.depth}手: ${choice.move}（${choice.score.toFixed(3)}）`).join(' / ')}`);
  console.log(`- 棋譜: ${row.position || '（初期局面）'}`);
}
