import { type Level, LEVELS, search } from '../engine/ai';
import type { GameState, Pos } from '../engine/game';
import { champion } from './champion';

export interface AiRequest {
  id: number;
  state: GameState;
  level: Level;
  recent: Pos[];
}

self.onmessage = (e: MessageEvent<AiRequest>) => {
  const { id, state, level, recent } = e.data;
  const result = search(state, { ...LEVELS[level], weights: champion?.weights, recent });
  self.postMessage({ id, move: result.move });
};
