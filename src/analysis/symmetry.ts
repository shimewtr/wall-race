// 盤面と手を、ルール上等価な対称形へ写す。
// AI の選択手・評価値が対称形でも一致するかを調べる実験で使う。

import { search, type SearchOptions, type SearchResult } from '../engine/ai';
import {
  type GameState,
  type Move,
  type Player,
  type Pos,
  SIZE,
  WALL_SLOTS,
  legalMoves,
  moveToString,
} from '../engine/game';

export type Symmetry = 'left-right' | 'swap-vertical';

const transformPlayer = (player: Player, symmetry: Symmetry): Player =>
  symmetry === 'swap-vertical' ? ((1 - player) as Player) : player;

export function transformPos(pos: Pos, symmetry: Symmetry): Pos {
  return symmetry === 'left-right'
    ? { x: SIZE - 1 - pos.x, y: pos.y }
    : { x: pos.x, y: SIZE - 1 - pos.y };
}

export function transformMove(move: Move, symmetry: Symmetry): Move {
  if (move.type === 'pawn') return { type: 'pawn', ...transformPos(move, symmetry) };
  return symmetry === 'left-right'
    ? { ...move, x: WALL_SLOTS - 1 - move.x }
    : { ...move, y: WALL_SLOTS - 1 - move.y };
}

export function transformState(state: GameState, symmetry: Symmetry): GameState {
  const hWalls = new Uint8Array(state.hWalls.length);
  const vWalls = new Uint8Array(state.vWalls.length);
  for (let y = 0; y < WALL_SLOTS; y++) {
    for (let x = 0; x < WALL_SLOTS; x++) {
      const source = y * WALL_SLOTS + x;
      const tx = symmetry === 'left-right' ? WALL_SLOTS - 1 - x : x;
      const ty = symmetry === 'swap-vertical' ? WALL_SLOTS - 1 - y : y;
      const target = ty * WALL_SLOTS + tx;
      hWalls[target] = state.hWalls[source];
      vWalls[target] = state.vWalls[source];
    }
  }

  const pawns: [Pos, Pos] =
    symmetry === 'left-right'
      ? [transformPos(state.pawns[0], symmetry), transformPos(state.pawns[1], symmetry)]
      : [transformPos(state.pawns[1], symmetry), transformPos(state.pawns[0], symmetry)];
  const wallsLeft: [number, number] =
    symmetry === 'left-right'
      ? [state.wallsLeft[0], state.wallsLeft[1]]
      : [state.wallsLeft[1], state.wallsLeft[0]];

  return {
    pawns,
    wallsLeft,
    hWalls,
    vWalls,
    turn: transformPlayer(state.turn, symmetry),
    ply: state.ply,
    winner: state.winner === null ? null : transformPlayer(state.winner, symmetry),
    draw: state.draw,
  };
}

const moveSet = (state: GameState) => legalMoves(state).map(moveToString).sort();

export interface SymmetryCheck {
  symmetry: Symmetry;
  expectedMove: string;
  actualMove: string;
  originalScore: number;
  transformedScore: number;
  scoreDelta: number;
  moveMatches: boolean;
  legalMovesMatch: boolean;
  originalNodes: number;
  transformedNodes: number;
}

export function checkSymmetry(
  state: GameState,
  options: SearchOptions,
  symmetry: Symmetry,
  original: SearchResult = search(state, options),
): SymmetryCheck {
  const transformedState = transformState(state, symmetry);
  const transformed = search(transformedState, options);
  const expectedMove = moveToString(transformMove(original.move, symmetry));
  const actualMove = moveToString(transformed.move);
  const expectedLegalMoves = legalMoves(state).map((move) => moveToString(transformMove(move, symmetry))).sort();

  return {
    symmetry,
    expectedMove,
    actualMove,
    originalScore: original.score,
    transformedScore: transformed.score,
    scoreDelta: transformed.score - original.score,
    moveMatches: expectedMove === actualMove,
    legalMovesMatch: JSON.stringify(expectedLegalMoves) === JSON.stringify(moveSet(transformedState)),
    originalNodes: original.nodes,
    transformedNodes: transformed.nodes,
  };
}
