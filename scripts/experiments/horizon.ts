// 同じ局面を複数の深さで探索し、読みが深くなると判断がどう変わるかを集計する。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/horizon.ts --games 20 --positions 120 --max-depth 4 --sample-every 4

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { recentCells, search } from '../../src/engine/ai';
import { type GameState, type Move, applyMove, initialState, isOver, moveToString } from '../../src/engine/game';
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
  console.log(`- 評価の符号反転: ${signChanges.length} / ${pairs.length}（${pct(signChanges.length, pairs.length)}）`);
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

console.log('\n## 深さ別の探索量');
for (let depth = 1; depth <= maxDepth; depth++) {
  console.log(`- ${depth}手読み: 平均 ${average(observations.map((row) => row.choices[depth - 1].nodes))} ノード`);
}

const lastChanges = observations
  .filter((row) => row.choices[maxDepth - 2].move !== row.choices[maxDepth - 1].move)
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
