import { describe, expect, it } from 'vitest';
import {
  type GameState,
  applyMove,
  initialState,
  isLegal,
  isLegalWall,
  legalMoves,
  moveFromString,
  moveToString,
  pawnMoves,
  shortestDistance,
} from './game';
import { search } from './ai';

const sortPos = (ps: { x: number; y: number }[]) =>
  ps.map((p) => `${p.x},${p.y}`).sort();

const place = (s: GameState, x: number, y: number, o: 'h' | 'v') =>
  applyMove(s, { type: 'wall', x, y, o });

describe('初期状態', () => {
  it('最短距離はお互い 8', () => {
    const s = initialState();
    expect(shortestDistance(s, 0)).toBe(8);
    expect(shortestDistance(s, 1)).toBe(8);
  });

  it('合法手は 駒 3 + 壁 128', () => {
    expect(legalMoves(initialState())).toHaveLength(3 + 128);
  });
});

describe('壁', () => {
  it('横壁は真上への移動をふさぐ', () => {
    // プレイヤー 0 は (4,8)。アンカー (4,7) の横壁で (4,7)-(4,8) の間をふさぐ
    const s = place(initialState(), 4, 7, 'h');
    s.turn = 0;
    expect(sortPos(pawnMoves(s))).toEqual(sortPos([{ x: 3, y: 8 }, { x: 5, y: 8 }]));
  });

  it('重なる壁・交差する壁は置けない', () => {
    const s = place(initialState(), 3, 3, 'h');
    expect(isLegalWall(s, 3, 3, 'h')).toBe(false);
    expect(isLegalWall(s, 2, 3, 'h')).toBe(false);
    expect(isLegalWall(s, 4, 3, 'h')).toBe(false);
    expect(isLegalWall(s, 3, 3, 'v')).toBe(false);
    expect(isLegalWall(s, 5, 3, 'h')).toBe(true);
    expect(isLegalWall(s, 3, 2, 'v')).toBe(true);
  });

  it('ゴールへの道を完全にふさぐ壁は置けない', () => {
    // 下段 y=7 と y=8 の間を横壁 4 枚で 0..7 列ふさぎ、最後の 8 列目を縦壁で囲う形を作る
    let s = initialState();
    for (const x of [0, 2, 4, 6]) s = place(s, x, 7, 'h');
    // 残る出口は x=8 の列。縦壁 (7,7) で (7,8)-(8,8) の間をふさぐとプレイヤー 0 が出られなくなる
    expect(isLegalWall(s, 7, 7, 'v')).toBe(false);
  });

  it('残り枚数が 0 なら置けない', () => {
    const s = initialState();
    s.wallsLeft[0] = 0;
    expect(isLegalWall(s, 0, 0, 'h')).toBe(false);
  });
});

describe('ジャンプ', () => {
  it('正面の相手を飛び越えられる', () => {
    const s = initialState();
    s.pawns = [{ x: 4, y: 5 }, { x: 4, y: 4 }];
    expect(sortPos(pawnMoves(s))).toContain('4,3');
    expect(sortPos(pawnMoves(s))).not.toContain('4,4');
  });

  it('相手の後ろが壁なら斜めに回り込む', () => {
    let s = initialState();
    s.pawns = [{ x: 4, y: 5 }, { x: 4, y: 4 }];
    s = place(s, 4, 3, 'h'); // (4,4) の上辺をふさぐ
    s.turn = 0;
    expect(sortPos(pawnMoves(s))).toEqual(sortPos([
      { x: 3, y: 4 }, { x: 5, y: 4 }, // 斜め
      { x: 3, y: 5 }, { x: 5, y: 5 }, { x: 4, y: 6 },
    ]));
  });
});

describe('勝敗', () => {
  it('ゴール行に着いたら勝ち', () => {
    const s = initialState();
    s.pawns[0] = { x: 0, y: 1 };
    const n = applyMove(s, { type: 'pawn', x: 0, y: 0 });
    expect(n.winner).toBe(0);
    expect(legalMoves(n)).toHaveLength(0);
  });
});

describe('棋譜表記', () => {
  it('往復できる', () => {
    for (const m of legalMoves(initialState())) {
      expect(moveFromString(moveToString(m))).toEqual(m);
    }
  });
});

describe('AI', () => {
  it('1 手で勝てるなら勝つ', () => {
    const s = initialState();
    s.pawns[0] = { x: 2, y: 1 };
    const r = search(s, { maxDepth: 2 });
    expect(r.move).toEqual({ type: 'pawn', x: 2, y: 0 });
  });

  it('相手が 1 手で勝つなら壁でふさぐ', () => {
    const s = initialState();
    s.pawns[1] = { x: 2, y: 7 };
    const r = search(s, { maxDepth: 2 });
    expect(r.move.type).toBe('wall');
    expect(isLegal(s, r.move)).toBe(true);
  });
});
