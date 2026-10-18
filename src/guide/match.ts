// 手引き AI と探索 AI を対戦させる。

import { type Weights, recentCells, search } from '../engine/ai';
import { type GameState, type Player, applyMove, initialState, isOver } from '../engine/game';
import { OPENING_PLIES, pickAt, rngFrom } from '../train/match';
import { type GuideParams, guideMove } from './policy';

export interface GuideJob {
  params: GuideParams;
  // 相手の探索 AI
  weights: Partial<Weights>;
  depth: number;
  guidePlayer: Player;
  seed: number;
}

// 手引き AI の得点（勝ち 1 / 引き分け 0.5 / 負け 0）
export function playGuide(job: GuideJob): number {
  const rng = rngFrom(job.seed);
  const history: GameState[] = [initialState()];
  let s = history[0];
  while (!isOver(s)) {
    const m =
      s.turn === job.guidePlayer
        ? guideMove(s, job.params)
        : search(s, {
            maxDepth: job.depth,
            weights: job.weights,
            margin: 1,
            pickFromTop: pickAt(s.ply, OPENING_PLIES),
            rng,
            recent: recentCells(history, s.turn),
          }).move;
    s = applyMove(s, m);
    history.push(s);
  }
  return s.winner === null ? 0.5 : s.winner === job.guidePlayer ? 1 : 0;
}
