import { describe, expect, it } from 'vitest';
import { search, type Weights } from '../engine/ai';
import { applyMove, initialState, legalMoves, moveFromString, moveToString, type GameState } from '../engine/game';
import { checkSymmetry, transformMove, transformState, type Symmetry } from './symmetry';

const asymmetricState = (): GameState => {
  let state = initialState();
  state = applyMove(state, { type: 'pawn', x: 3, y: 8 });
  state = applyMove(state, { type: 'wall', x: 2, y: 2, o: 'h' });
  state = applyMove(state, { type: 'wall', x: 5, y: 5, o: 'v' });
  return state;
};

const legalNames = (state: GameState) => legalMoves(state).map(moveToString).sort();

const replay = (moves: string): GameState =>
  moves.split(' ').reduce((state, text) => {
    const move = moveFromString(text);
    if (!move) throw new Error(`invalid move: ${text}`);
    return applyMove(state, move);
  }, initialState());

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

it('wallChain を含む探索評価が左右の候補走査順に依存しない', () => {
  const state = replay('e2 e8 d2 f8 d3 e8 d4 e7 d5 e6 d6 e5 d7 e4 d8 c8h e8 e8h f8 f7v f7 e3 f6 e2');
  const weights: Weights = {
    pathDiff: 10,
    rowDiff: -1.671,
    wallDiff: 19.743,
    myWalls: 4.9,
    urgency: 675.456,
    wallsWhenClose: 54.977,
    center: -1.119,
    adjacent: 2.913,
    lockedRace: 27.437,
    wallThreat: 5.861,
    netWallThreat: 3.906,
    wallChain: 2.938,
    routeDiversity: -1.807,
  };
  const original = search(state, { maxDepth: 3, weights, repeatPenalty: 0 });
  const result = checkSymmetry(state, { maxDepth: 3, weights, repeatPenalty: 0 }, 'left-right', original);
  expect(result.scoreDelta).toBeCloseTo(0, 10);
});
