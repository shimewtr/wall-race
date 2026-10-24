import { describe, expect, it } from 'vitest';
import { search } from '../engine/ai';
import { applyMove, initialState, legalMoves, moveToString, type GameState } from '../engine/game';
import { checkSymmetry, transformMove, transformState, type Symmetry } from './symmetry';

const asymmetricState = (): GameState => {
  let state = initialState();
  state = applyMove(state, { type: 'pawn', x: 3, y: 8 });
  state = applyMove(state, { type: 'wall', x: 2, y: 2, o: 'h' });
  state = applyMove(state, { type: 'wall', x: 5, y: 5, o: 'v' });
  return state;
};

const legalNames = (state: GameState) => legalMoves(state).map(moveToString).sort();

describe.each<Symmetry>(['left-right', 'swap-vertical'])('symmetry: %s', (symmetry) => {
  it('2回変換すると元の盤面に戻る', () => {
    const state = asymmetricState();
    expect(transformState(transformState(state, symmetry), symmetry)).toEqual(state);
  });

  it('合法手の集合を保つ', () => {
    const state = asymmetricState();
    const expected = legalMoves(state).map((move) => moveToString(transformMove(move, symmetry))).sort();
    expect(legalNames(transformState(state, symmetry))).toEqual(expected);
  });

  it('静的評価と探索評価を保つ', () => {
    const state = asymmetricState();
    const options = { maxDepth: 1, repeatPenalty: 0 };
    const original = search(state, options);
    const result = checkSymmetry(state, options, symmetry, original);
    expect(result.legalMovesMatch).toBe(true);
    expect(result.scoreDelta).toBeCloseTo(0, 10);
  });
});
