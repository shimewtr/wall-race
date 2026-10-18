import { type Level, LEVELS, search } from '../engine/ai';
import type { GameState, Move, Pos } from '../engine/game';
import { champion } from './champion';

// AI の手を考える依頼と、形勢を判断する依頼の 2 種類
export type AiRequest =
  | { kind: 'move'; id: number; state: GameState; level: Level; recent: Pos[] }
  | { kind: 'eval'; id: number; state: GameState };

export type AiResponse = { kind: 'move'; id: number; move: Move } | { kind: 'eval'; id: number; score: number };

// 形勢判断の読みの深さ（「つよい」と同じ）
export const EVAL_DEPTH = 3;

self.onmessage = (e: MessageEvent<AiRequest>) => {
  const req = e.data;
  if (req.kind === 'move') {
    const result = search(req.state, { ...LEVELS[req.level], weights: champion?.weights, recent: req.recent });
    self.postMessage({ kind: 'move', id: req.id, move: result.move } satisfies AiResponse);
  } else {
    const result = search(req.state, { maxDepth: EVAL_DEPTH, weights: champion?.weights });
    self.postMessage({ kind: 'eval', id: req.id, score: result.score } satisfies AiResponse);
  }
};
