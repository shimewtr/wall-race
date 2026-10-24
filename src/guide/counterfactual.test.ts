import { describe, expect, it } from 'vitest';
import { moveToString } from '../engine/game';
import { replayPosition } from './counterfactual';

describe('replayPosition', () => {
  it('発見ログの棋譜から局面を再現する', () => {
    const history = replayPosition({
      ply: 2,
      guidePlayer: 0,
      pawns: ['e2', 'e8'],
      wallsLeft: [10, 10],
      hWalls: [],
      vWalls: [],
      moves: 'e2 e8',
    });

    expect(history).toHaveLength(3);
    expect(history.at(-1)!.turn).toBe(0);
    expect(history.at(-1)!.pawns.map((pawn) => moveToString({ type: 'pawn', ...pawn }))).toEqual(['e2', 'e8']);
  });
});
