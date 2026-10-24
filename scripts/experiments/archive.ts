// 3手読みと4手読みで判断が割れる局面を、重点学習候補のアーカイブとして保存する。
// GitHub Actions の Experiment で実行し、experiment-output.json をartifactから取得する。

import { writeFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { recentCells, search } from '../../src/engine/ai';
import {
  type GameState,
  type Move,
  applyMove,
  initialState,
  isOver,
  moveToString,
  shortestDistance,
} from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '80' },
    positions: { type: 'string', default: '1200' },
    limit: { type: 'string', default: '120' },
    'sample-every': { type: 'string', default: '4' },
    seed: { type: 'string', default: '97000000' },
    output: { type: 'string', default: 'experiment-output.json' },
  },
});

const champion = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'));
const games = Number(args.games);
const maxPositions = Number(args.positions);
const limit = Number(args.limit);
const sampleEvery = Number(args['sample-every']);
const seed = Number(args.seed);

const phaseAt = (ply: number) => ply < 20 ? 'opening' : ply < 40 ? 'middle' : 'endgame';
const transition = (a: Move, b: Move) => {
  if (a.type !== b.type) return `${a.type}-to-${b.type}`;
  if (a.type === 'wall' && b.type === 'wall' && a.o !== b.o) return 'wall-orientation';
  return a.type === 'pawn' ? 'pawn-target' : 'wall-target';
};
const stateKey = (state: GameState) => [
  state.turn,
  ...state.pawns.flatMap((pawn) => [pawn.x, pawn.y]),
  ...state.wallsLeft,
  ...state.hWalls,
  ...state.vWalls,
].join(',');

interface Candidate {
  phase: string;
  transition: string;
  game: number;
  history: string[];
  state: {
    pawns: GameState['pawns'];
    wallsLeft: GameState['wallsLeft'];
    hWalls: number[];
    vWalls: number[];
    turn: GameState['turn'];
    ply: number;
  };
  myDistance: number;
  opponentDistance: number;
  depth3: { move: string; score: number; nodes: number };
  depth4: { move: string; score: number; nodes: number };
}

const candidates: Candidate[] = [];
const seen = new Set<string>();
let examined = 0;
for (let game = 0; game < games && examined < maxPositions; game++) {
  const rng = rngFrom(seed + game);
  const states: GameState[] = [initialState()];
  const history: string[] = [];
  let state = states[0];
  while (!isOver(state) && examined < maxPositions) {
    if (state.ply % sampleEvery === game % sampleEvery) {
      examined++;
      const depth3 = search(state, { maxDepth: 3, weights: champion.weights, repeatPenalty: 0 });
      const depth4 = search(state, { maxDepth: 4, weights: champion.weights, repeatPenalty: 0 });
      if (moveToString(depth3.move) !== moveToString(depth4.move)) {
        const key = stateKey(state);
        if (!seen.has(key)) {
          seen.add(key);
          candidates.push({
            phase: phaseAt(state.ply),
            transition: transition(depth3.move, depth4.move),
            game: game + 1,
            history: [...history],
            state: {
              pawns: [{ ...state.pawns[0] }, { ...state.pawns[1] }],
              wallsLeft: [...state.wallsLeft],
              hWalls: [...state.hWalls],
              vWalls: [...state.vWalls],
              turn: state.turn,
              ply: state.ply,
            },
            myDistance: shortestDistance(state, state.turn),
            opponentDistance: shortestDistance(state, (1 - state.turn) as 0 | 1),
            depth3: { move: moveToString(depth3.move), score: depth3.score, nodes: depth3.nodes },
            depth4: { move: moveToString(depth4.move), score: depth4.score, nodes: depth4.nodes },
          });
        }
      }
    }
    const move = search(state, {
      maxDepth: depthAt(state.ply, 3, OPENING_PLIES),
      weights: champion.weights,
      margin: 1,
      pickFromTop: pickAt(state.ply, OPENING_PLIES),
      rng,
      recent: recentCells(states, state.turn),
    }).move;
    history.push(moveToString(move));
    state = applyMove(state, move);
    states.push(state);
  }
}

// 各フェーズから順番に取り、終盤の長い対局だけに偏らないようにする。
const phaseQueues = ['opening', 'middle', 'endgame'].map((phase) => candidates.filter((row) => row.phase === phase));
const selected: Candidate[] = [];
while (selected.length < limit && phaseQueues.some((queue) => queue.length > 0)) {
  for (const queue of phaseQueues) {
    const item = queue.shift();
    if (item) selected.push(item);
    if (selected.length >= limit) break;
  }
}

const archive = {
  version: 1,
  championGeneration: champion.generation,
  source: { seed, games, examined, sampleEvery },
  selection: 'depth3-depth4-move-disagreement, phase-round-robin',
  positions: selected,
};
writeFileSync(args.output, `${JSON.stringify(archive, null, 2)}\n`);

console.log(`# 第 ${champion.generation} 世代の決定的局面アーカイブ`);
console.log();
console.log(`調査: ${examined}局面 / 判断差: ${candidates.length} / 保存: ${selected.length}`);
console.log(`出力: ${args.output}`);
for (const phase of ['opening', 'middle', 'endgame']) {
  console.log(`- ${phase}: ${selected.filter((row) => row.phase === phase).length}`);
}
const transitions = new Map<string, number>();
for (const row of selected) transitions.set(row.transition, (transitions.get(row.transition) ?? 0) + 1);
console.log('- 遷移: ' + [...transitions].sort((a, b) => b[1] - a[1]).map(([name, count]) => `${name} ${count}`).join(' / '));
console.log('- このファイルはまだTrainから参照しない。開始局面比率10%・20%の独立比較に合格した場合だけ学習へ接続する。');
