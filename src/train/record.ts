// 1 局を指しながら、手ごとに「どんな状況で何をしたか」を記録する。手引き作りの分析用。

import { type Weights, candidateWalls, recentCells, search } from '../engine/ai';
import { type GameState, type Move, type Player, applyMove, initialState, isOver, moveToString, shortestDistance } from '../engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from './match';

export type WallKind = 'block' | 'near' | 'guard';

export interface MoveRecord {
  ply: number;
  player: Player;
  move: string;
  type: Move['type'];
  // 壁の種類: block = 相手の最短経路をふさぐ / near = コマに接する / guard = 自分の経路沿いの守り
  kind?: WallKind;
  myDist: number;
  opDist: number;
  myDistAfter: number;
  opDistAfter: number;
  myWalls: number;
  opWalls: number;
  // 指した側から見た探索の評価値（1 歩 = 10 点）
  score: number;
}

export interface GameRecord {
  winner: Player | null;
  moves: MoveRecord[];
}

export interface RecordJob {
  weights: Partial<Weights>;
  depth: number;
  seed: number;
}

export function playRecorded(job: RecordJob): GameRecord {
  const rng = rngFrom(job.seed);
  const history: GameState[] = [initialState()];
  let s = history[0];
  const moves: MoveRecord[] = [];
  while (!isOver(s)) {
    const me = s.turn;
    const op = (1 - me) as Player;
    const result = search(s, {
      maxDepth: depthAt(s.ply, job.depth, OPENING_PLIES),
      weights: job.weights,
      margin: 1,
      pickFromTop: pickAt(s.ply, OPENING_PLIES),
      rng,
      recent: recentCells(history, me),
    });
    const m = result.move;
    let kind: WallKind | undefined;
    if (m.type === 'wall') {
      const same = (w: Move) => w.type === 'wall' && w.x === m.x && w.y === m.y && w.o === m.o;
      const touchesPawn = s.pawns.some((p) => (m.x === p.x || m.x === p.x - 1) && (m.y === p.y || m.y === p.y - 1));
      kind = candidateWalls(s).some(same) ? 'block' : touchesPawn ? 'near' : 'guard';
    }
    const next = applyMove(s, m);
    moves.push({
      ply: s.ply,
      player: me,
      move: moveToString(m),
      type: m.type,
      kind,
      myDist: shortestDistance(s, me),
      opDist: shortestDistance(s, op),
      myDistAfter: shortestDistance(next, me),
      opDistAfter: shortestDistance(next, op),
      myWalls: s.wallsLeft[me],
      opWalls: s.wallsLeft[op],
      score: Math.round(result.score * 10) / 10,
    });
    s = next;
    history.push(s);
  }
  return { winner: s.winner, moves };
}
