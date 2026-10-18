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
import { candidateWalls, features, recentCells, search } from './ai';
import { rngFrom } from '../train/match';
import { guideMove } from '../guide/policy';

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

describe('行ったり来たり防止', () => {
  it('recentCells は自分の手番で直近にいたマスを返す（現在地は除く）', () => {
    const h: GameState[] = [initialState()];
    const play = (m: Parameters<typeof applyMove>[1]) => h.push(applyMove(h[h.length - 1], m));
    play({ type: 'pawn', x: 3, y: 8 }); // 先手 e1 → d1
    play({ type: 'pawn', x: 4, y: 1 });
    play({ type: 'pawn', x: 4, y: 8 }); // 先手 d1 → e1（戻った）
    play({ type: 'pawn', x: 4, y: 2 });
    expect(recentCells(h, 0)).toEqual([{ x: 3, y: 8 }]);
  });

  const flat = { pathDiff: 0, wallDiff: 0, myWalls: 0, urgency: 0 };

  it('同点の手なら直近にいたマスへは戻らない', () => {
    // 評価をすべて 0 にして全手同点にする。減点がなければ並び順で先頭の前進 (e2) を選ぶ
    const s = initialState();
    s.wallsLeft[0] = 0;
    expect(search(s, { maxDepth: 1, weights: flat }).move).toEqual({ type: 'pawn', x: 4, y: 7 });
    const r = search(s, { maxDepth: 1, weights: flat, recent: [{ x: 4, y: 7 }] });
    expect(r.move.type).toBe('pawn');
    expect(r.move).not.toEqual({ type: 'pawn', x: 4, y: 7 });
  });

  it('1 歩得するなら直近にいたマスへでも戻る', () => {
    const s = initialState();
    s.wallsLeft[0] = 0;
    const r = search(s, { maxDepth: 1, recent: [{ x: 4, y: 7 }] });
    expect(r.move).toEqual({ type: 'pawn', x: 4, y: 7 });
  });
});

describe('ランダム性（margin）', () => {
  const flat = { pathDiff: 0, wallDiff: 0, myWalls: 0, urgency: 0 };

  it('同点の手が複数あれば乱数によって違う手を選ぶ', () => {
    const s = initialState();
    s.wallsLeft[0] = 0;
    const picked = new Set(
      Array.from({ length: 20 }, (_, i) => {
        const r = search(s, { maxDepth: 1, weights: flat, margin: 1, rng: rngFrom(i + 1) });
        return moveToString(r.move);
      }),
    );
    expect(picked.size).toBeGreaterThan(1);
  });

  it('margin を超えて悪い手は選ばない', () => {
    const s = initialState();
    s.wallsLeft[0] = 0;
    for (let i = 0; i < 20; i++) {
      const r = search(s, { maxDepth: 2, margin: 1, rng: rngFrom(i + 1) });
      expect(r.move).toEqual({ type: 'pawn', x: 4, y: 7 });
    }
  });
});

describe('特徴量 lockedRace', () => {
  it('相手の壁が尽きていて自分が先着できるなら +1', () => {
    const s = initialState();
    s.wallsLeft[1] = 0;
    expect(features(s).lockedRace).toBe(1); // 手番の先手は同じ 8 歩なら先に着く
  });

  it('自分の壁が尽きていて相手が先着できるなら -1', () => {
    const s = initialState();
    s.wallsLeft[0] = 0;
    s.pawns[1] = { x: 4, y: 2 }; // 相手は残り 6 歩、自分は 8 歩
    expect(features(s).lockedRace).toBe(-1);
  });

  it('お互い壁が残っていれば 0', () => {
    expect(features(initialState()).lockedRace).toBe(0);
  });
});

describe('壁の候補（wide）', () => {
  const key = (m: { x: number; y: number; o?: string }) => `${m.x},${m.y},${m.o}`;

  it('wide は通常の候補をすべて含み、さらに増える', () => {
    const s = initialState();
    const narrow = candidateWalls(s).map(key);
    const wide = candidateWalls(s, true).map(key);
    for (const k of narrow) expect(wide).toContain(k);
    expect(wide.length).toBeGreaterThan(narrow.length);
  });

  it('自分の最短経路を伸ばす壁は守りの壁として加えない', () => {
    // 先手は e1 から真上（e 列）に進むのが最短。e1-e2 の間をふさぐ横壁は自分の経路を伸ばす
    const s = initialState();
    const extra = candidateWalls(s, true).filter((m) => !candidateWalls(s).some((n) => key(n) === key(m)));
    for (const m of extra) {
      if (m.type !== 'wall') continue;
      const touchesPawn = s.pawns.some((p) => (m.x === p.x || m.x === p.x - 1) && (m.y === p.y || m.y === p.y - 1));
      if (!touchesPawn) expect(shortestDistance(applyMove(s, m), 0)).toBe(8);
    }
  });
});

describe('手引き AI', () => {
  it('先着できる局面では前進する', () => {
    const m = guideMove(initialState());
    expect(m).toEqual({ type: 'pawn', x: 4, y: 7 });
  });

  it('負けている局面では相手の道をふさぐ壁を置く', () => {
    const s = initialState();
    s.pawns[1] = { x: 1, y: 5 }; // 相手は残り 3 歩、自分は 8 歩（別の列なので、ふさいでも自分の道は伸びない）
    const m = guideMove(s);
    expect(m.type).toBe('wall');
    expect(shortestDistance(applyMove(s, m), 1)).toBeGreaterThan(3);
  });
});
