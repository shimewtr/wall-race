import { describe, expect, it } from 'vitest';
import { initialState } from '../engine/game';
import { DEFAULT_WEIGHTS } from '../engine/ai';
import { playMatch } from './match';

describe('playMatch startHistory', () => {
  it('終局済みの中間状態ならその勝者を返す', () => {
    const state = initialState();
    state.winner = 1;
    expect(playMatch({
      weights: [DEFAULT_WEIGHTS, DEFAULT_WEIGHTS],
      depth: 1,
      margin: 0,
      openingPlies: 0,
      seed: 1,
      startHistory: [state],
    })).toBe(1);
  });

  it('空の履歴を拒否する', () => {
    expect(() => playMatch({
      weights: [DEFAULT_WEIGHTS, DEFAULT_WEIGHTS],
      depth: 1,
      margin: 0,
      openingPlies: 0,
      seed: 1,
      startHistory: [],
    })).toThrow('startHistory must not be empty');
  });
});
