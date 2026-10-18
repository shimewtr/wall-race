import { describe, expect, it } from 'vitest';
import {
  type GameState,
  applyMove,
  cloneState,
  goalDistances,
  makeMove,
  shortestPath,
  unmakeMove,
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

describe('高速化した処理が元の処理と一致する', () => {
  // 壁をランダムに置いた局面をたくさん作る
  const randomStates = (n: number) => {
    const rng = rngFrom(42);
    const states: GameState[] = [];
    for (let i = 0; i < n; i++) {
      let s = initialState();
      s.pawns = [
        { x: Math.floor(rng() * 9), y: 1 + Math.floor(rng() * 7) },
        { x: Math.floor(rng() * 9), y: 1 + Math.floor(rng() * 7) },
      ];
      const walls = Math.floor(rng() * 16);
      for (let k = 0; k < walls; k++) {
        const m = { type: 'wall' as const, x: Math.floor(rng() * 8), y: Math.floor(rng() * 8), o: rng() < 0.5 ? ('h' as const) : ('v' as const) };
        s.turn = (k % 2) as 0 | 1;
        if (isLegal(s, m)) s = applyMove(s, m);
      }
      s.turn = (i % 2) as 0 | 1;
      states.push(s);
    }
    return states;
  };

  it('shortestDistance は最短経路の長さと同じ', () => {
    for (const s of randomStates(300)) {
      for (const p of [0, 1] as const) {
        const path = shortestPath(s, p);
        expect(shortestDistance(s, p)).toBe(path ? path.length - 1 : -1);
      }
    }
  });

  it('goalDistances は各マスにコマを置いたときの最短距離と同じ', () => {
    for (const s of randomStates(40)) {
      const p = s.turn;
      const map = Array.from(goalDistances(s, p));
      for (let y = 0; y < 9; y++) {
        for (let x = 0; x < 9; x++) {
          const t = cloneState(s);
          t.pawns[p] = { x, y };
          expect(map[y * 9 + x]).toBe(shortestDistance(t, p));
        }
      }
    }
  });

  it('壁の候補は isLegalWall で判定した場合と同じ', () => {
    // 参照実装: 相手の最短経路の各歩をふさぐ壁を isLegalWall で判定する（高速化前のやり方）
    const reference = (s: GameState) => {
      if (s.wallsLeft[s.turn] <= 0) return [];
      const path = shortestPath(s, (1 - s.turn) as 0 | 1)!;
      const keys: string[] = [];
      const add = (x: number, y: number, o: 'h' | 'v') => {
        const k = `${x},${y},${o}`;
        if (x >= 0 && y >= 0 && x < 8 && y < 8 && !keys.includes(k) && isLegalWall(s, x, y, o)) keys.push(k);
      };
      for (let i = 0; i + 1 < path.length; i++) {
        const [a, b] = [path[i], path[i + 1]];
        if (a.x === b.x) {
          add(a.x, Math.min(a.y, b.y), 'h');
          add(a.x - 1, Math.min(a.y, b.y), 'h');
        } else {
          add(Math.min(a.x, b.x), a.y, 'v');
          add(Math.min(a.x, b.x), a.y - 1, 'v');
        }
      }
      return keys.sort();
    };
    for (const s of randomStates(300)) {
      const fast = candidateWalls(s).map((m) => (m.type === 'wall' ? `${m.x},${m.y},${m.o}` : '')).sort();
      expect(fast).toEqual(reference(s));
      for (const m of candidateWalls(s, true)) {
        if (m.type === 'wall') expect(isLegalWall(s, m.x, m.y, m.o)).toBe(true);
      }
    }
  });

  it('makeMove → unmakeMove で元の盤面に戻る', () => {
    for (const s of randomStates(100)) {
      const before = JSON.stringify({ ...s, hWalls: [...s.hWalls], vWalls: [...s.vWalls] });
      for (const m of legalMoves(s)) {
        const undo = makeMove(s, m);
        unmakeMove(s, m, undo);
      }
      expect(JSON.stringify({ ...s, hWalls: [...s.hWalls], vWalls: [...s.vWalls] })).toBe(before);
    }
  });
});
