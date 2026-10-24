// 高信頼なのに勝敗を外した局面を、同条件で正しく予測した局面と比較する。
// 重い自己対戦と4手読みを含むため、GitHub Actions の Experiment で実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { recentCells, search, withDefaults } from '../../src/engine/ai';
import { type GameState, type Move, type Player, applyMove, initialState, moveFromString, moveToString } from '../../src/engine/game';
import { type Calibration, calibrationUncertainty, winRate } from '../../src/train/calibrate';
import type { GameRecord, MoveRecord } from '../../src/train/record';
import { RecordPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '800' },
    limit: { type: 'string', default: '120' },
    depth: { type: 'string', default: '3' },
    seed: { type: 'string', default: '99400000' },
  },
});

const root = join(import.meta.dirname, '..', '..');
const champion = JSON.parse(readFileSync(join(root, 'training', 'champion.json'), 'utf8')) as { generation: number; weights: Record<string, number> };
const calibration = JSON.parse(readFileSync(join(root, 'training', 'calibration.json'), 'utf8')) as Calibration;
const games = Number(args.games);
const limit = Number(args.limit);
const depth = Number(args.depth);
const seed = Number(args.seed);

const pool = new RecordPool();
const records = await pool.runAll(Array.from({ length: games }, (_, i) => ({
  weights: withDefaults(champion.weights),
  depth,
  seed: seed + i,
})));
await pool.close();

type Phase = 'opening' | 'middle' | 'endgame';
type WallBand = 'low' | 'middle' | 'high';
interface PositionRef {
  gameIndex: number;
  moveIndex: number;
  seed: number;
  predicted: number;
  confidence: number;
  uncertainty: number;
  won: boolean;
  favoredPlayer: Player;
  phase: Phase;
  walls: WallBand;
  move: MoveRecord;
}

const phaseOf = (move: MoveRecord): Phase => move.ply < 12 ? 'opening' : Math.min(move.myDist, move.opDist) <= 4 ? 'endgame' : 'middle';
const wallBandOf = (move: MoveRecord): WallBand => {
  const walls = move.myWalls + move.opWalls;
  return walls <= 6 ? 'low' : walls <= 13 ? 'middle' : 'high';
};
const ref = (game: GameRecord, gameIndex: number, move: MoveRecord, moveIndex: number): PositionRef => {
  const u = calibrationUncertainty(move);
  const predicted = winRate(move.score, calibration, u);
  const won = game.winner === move.player;
  return {
    gameIndex,
    moveIndex,
    seed: seed + gameIndex,
    predicted,
    confidence: Math.max(predicted, 1 - predicted),
    uncertainty: u,
    won,
    favoredPlayer: (predicted >= 0.5 ? move.player : 1 - move.player) as Player,
    phase: phaseOf(move),
    walls: wallBandOf(move),
    move,
  };
};

const all = records.flatMap((game, gameIndex) => game.moves.map((move, moveIndex) => ref(game, gameIndex, move, moveIndex)));
const wrongByGame = new Map<number, PositionRef[]>();
for (const item of all) {
  const wrong = (item.predicted >= 0.8 && !item.won) || (item.predicted <= 0.2 && item.won);
  if (!wrong) continue;
  const group = wrongByGame.get(item.gameIndex) ?? [];
  group.push(item);
  wrongByGame.set(item.gameIndex, group);
}
const errors = [...wrongByGame.values()]
  .map((items) => items.sort((a, b) => Math.abs(b.predicted - Number(b.won)) - Math.abs(a.predicted - Number(a.won)))[0])
  .sort((a, b) => b.confidence - a.confidence)
  .slice(0, limit);

// 同じフェーズ・壁量を優先し、不確実性・手数・自信度が近い正解局面を対照にする。
const correct = all.filter((item) => item.confidence >= 0.8 && ((item.predicted >= 0.5) === item.won));
const usedControls = new Set<string>();
const controls = errors.map((error) => {
  let best: PositionRef | null = null;
  let bestCost = Infinity;
  for (const item of correct) {
    const key = `${item.gameIndex}:${item.moveIndex}`;
    if (usedControls.has(key) || item.gameIndex === error.gameIndex) continue;
    const cost =
      (item.phase === error.phase ? 0 : 20) +
      (item.walls === error.walls ? 0 : 10) +
      Math.abs(item.uncertainty - error.uncertainty) * 5 +
      Math.abs(item.move.ply - error.move.ply) / 20 +
      Math.abs(item.confidence - error.confidence);
    if (cost < bestCost) {
      best = item;
      bestCost = cost;
    }
  }
  if (!best) throw new Error('not enough matched controls');
  usedControls.add(`${best.gameIndex}:${best.moveIndex}`);
  return best;
});

const replay = (record: GameRecord, moveIndex: number) => {
  const history: GameState[] = [initialState()];
  for (let i = 0; i < moveIndex; i++) {
    const move = moveFromString(record.moves[i].move);
    if (!move) throw new Error(`cannot parse move: ${record.moves[i].move}`);
    history.push(applyMove(history.at(-1)!, move));
  }
  return history;
};
const transition = (a: Move, b: Move) => {
  if (a.type !== b.type) return `${a.type}-to-${b.type}`;
  if (a.type === 'wall' && b.type === 'wall' && a.o !== b.o) return 'wall-orientation';
  return a.type === 'pawn' ? 'pawn-target' : 'wall-target';
};
const probabilityFor = (move: MoveRecord, player: Player) => {
  const mover = winRate(move.score, calibration, calibrationUncertainty(move));
  return move.player === player ? mover : 1 - mover;
};

const analyze = (item: PositionRef) => {
  const record = records[item.gameIndex];
  const history = replay(record, item.moveIndex);
  const state = history.at(-1)!;
  const options = { weights: champion.weights, recent: recentCells(history, state.turn) };
  const shallow = search(state, { ...options, maxDepth: 3 });
  const deep = search(state, { ...options, maxDepth: 4 });
  let flipIndex = -1;
  for (let i = item.moveIndex + 1; i < record.moves.length; i++) {
    if (probabilityFor(record.moves[i], item.favoredPlayer) < 0.5) {
      flipIndex = i;
      break;
    }
  }
  const trigger = flipIndex > item.moveIndex ? record.moves[flipIndex - 1] : null;
  return {
    seed: item.seed,
    game: item.gameIndex + 1,
    ply: item.move.ply,
    phase: item.phase,
    walls: item.walls,
    predicted: item.predicted,
    confidence: item.confidence,
    uncertainty: item.uncertainty,
    eventualWinner: record.winner,
    favoredPlayer: item.favoredPlayer,
    actualMove: item.move.move,
    history: record.moves.slice(0, item.moveIndex).map((move) => move.move),
    state: {
      pawns: state.pawns.map((pawn) => ({ ...pawn })),
      wallsLeft: [...state.wallsLeft],
      hWalls: [...state.hWalls],
      vWalls: [...state.vWalls],
      turn: state.turn,
      ply: state.ply,
    },
    depth3: { move: moveToString(shallow.move), score: shallow.score, nodes: shallow.nodes },
    depth4: { move: moveToString(deep.move), score: deep.score, nodes: deep.nodes },
    depthChanged: moveToString(shallow.move) !== moveToString(deep.move),
    depthTransition: transition(shallow.move, deep.move),
    pliesToFlip: flipIndex < 0 ? null : flipIndex - item.moveIndex,
    trigger: trigger ? { move: trigger.move, type: trigger.type, kind: trigger.kind ?? null } : null,
  };
};

const errorAnalysis = errors.map(analyze);
const controlAnalysis = controls.map(analyze);
const pct = (count: number, total: number) => total === 0 ? '-' : `${(count / total * 100).toFixed(1)}%`;
const countBy = <T>(items: T[], value: (item: T) => string) => Object.fromEntries(
  [...items.reduce((map, item) => map.set(value(item), (map.get(value(item)) ?? 0) + 1), new Map<string, number>())]
    .sort((a, b) => b[1] - a[1]),
);

console.log(`# 第${champion.generation}世代の高信頼誤予測分析`);
console.log(`自己対戦${games}局 / 高信頼誤予測のある対局${wrongByGame.size}局 / 分析${errorAnalysis.length}局面`);
console.log(`3→4手の変更: 誤予測 ${errorAnalysis.filter((item) => item.depthChanged).length}/${errorAnalysis.length}（${pct(errorAnalysis.filter((item) => item.depthChanged).length, errorAnalysis.length)}）、対照 ${controlAnalysis.filter((item) => item.depthChanged).length}/${controlAnalysis.length}（${pct(controlAnalysis.filter((item) => item.depthChanged).length, controlAnalysis.length)}）`);
console.log(`誤予測のフェーズ: ${JSON.stringify(countBy(errorAnalysis, (item) => item.phase))}`);
console.log(`誤予測の壁量: ${JSON.stringify(countBy(errorAnalysis, (item) => item.walls))}`);
console.log(`深さ変更の内訳: ${JSON.stringify(countBy(errorAnalysis.filter((item) => item.depthChanged), (item) => item.depthTransition))}`);
console.log(`形勢を初めて反転させた手: ${JSON.stringify(countBy(errorAnalysis.filter((item) => item.trigger), (item) => `${item.trigger!.type}:${item.trigger!.kind ?? '-'}`))}`);
const flips = errorAnalysis.flatMap((item) => item.pliesToFlip === null ? [] : [item.pliesToFlip]);
console.log(`反転までの手数: 平均${(flips.reduce((sum, value) => sum + value, 0) / flips.length).toFixed(1)}手 / ${flips.length}局面で観測`);

writeFileSync('experiment-output.json', `${JSON.stringify({
  version: 1,
  championGeneration: champion.generation,
  source: { games, limit, depth, seed },
  errors: errorAnalysis,
  controls: controlAnalysis,
}, null, 2)}\n`);
console.log('再現可能な局面と対照群をexperiment-output.jsonへ保存。');
