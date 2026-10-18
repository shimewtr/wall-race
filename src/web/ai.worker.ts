import { type Level, LEVELS, search } from '../engine/ai';
import type { GameState } from '../engine/game';

export interface AiRequest {
  id: number;
  state: GameState;
  level: Level;
}

self.onmessage = (e: MessageEvent<AiRequest>) => {
  const { id, state, level } = e.data;
  const result = search(state, LEVELS[level]);
  self.postMessage({ id, move: result.move });
};
