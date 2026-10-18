// 「手引き」をそのまま実行するだけの AI。先読みはせず、人間が盤面で確かめられることだけで手を決める。
// この AI がどれだけ強いかで、手引きの出来を測る（手引きを書き換えたらここも合わせて書き換える）。
//
// 手引き（初版）:
//   1. 自分と相手の「ゴールまでの残り歩数」を数える（壁を迂回する最短の歩数）
//   2. 自分の手番なので、自分の歩数 ≦ 相手の歩数 なら「このままなら先に着く」。前進する
//   3. 先に着けないなら壁を置く。置くのは相手の最短ルートをふさぐ壁のうち、
//      「相手が遠回りになる歩数 − 自分が遠回りになる歩数」がいちばん大きいもの
//   4. その差が minNetGain 歩に満たなければ、壁は置かずに前進する
//   5. 前進は自分の最短ルートに沿って 1 歩。同じなら盤の中央寄りを選ぶ
//
// 第 2 版（lookahead）: 1〜5 の代わりに、候補の手（前進と、相手の最短ルートをふさぐ壁）それぞれについて
//   「その手のあと、相手が 1〜5 のやり方で最善の邪魔（または前進）をしてきたら、歩数の差はどうなるか」
//   を確かめ、差がいちばん良くなる手を選ぶ。同じなら壁を温存できる前進を選ぶ。
// 第 3 版（lookahead: 2）: 相手も第 2 版と同じように考えて返してくる、と想定する。

import { candidateWalls, maxDetour } from '../engine/ai';
import { type GameState, type Move, type Player, applyMove, pawnMoves, shortestDistance } from '../engine/game';

export interface GuideParams {
  // 壁を置くのは「相手の増加 − 自分の増加」がこの歩数以上のときだけ
  minNetGain: number;
  // 自分がリードしていても、リードがこの歩数以下で相手がゴール間近（holdUntil 歩以内）なら壁で突き放す
  defendLead: number;
  holdUntil: number;
  // 相手の残り歩数がこれより多いうちは、負けていても壁を温存する（序盤に使い切らない）
  saveWallsWhileOpponentFarther: number;
  // 相手の応手を何段階まで想定するか。0 = 第 1 版、1 = 第 2 版、2 = 第 3 版
  lookahead?: number;
  // 形勢を比べるとき、残り壁 1 枚の差を何歩分と数えるか（0 なら歩数の差だけで比べる）
  wallValue?: number;
  // 返しのあと「お互いに壁 1 枚で相手を何歩遠回りさせられるか」の差を、この倍率で形勢に足す
  threatValue?: number;
}

export const GUIDE_V1: GuideParams = {
  minNetGain: 1,
  defendLead: -1,
  holdUntil: 0,
  saveWallsWhileOpponentFarther: 99,
};

export const GUIDE_V2: GuideParams = { ...GUIDE_V1, lookahead: 1 };
export const GUIDE_V3: GuideParams = { ...GUIDE_V1, lookahead: 2 };
// docs/guide.md に書いている手順（第 2 版）。
// 壁の枚数を歩数に換算して形勢に足す版（wallValue）も試したが、序盤をばらけさせた 400 局の比較で
// 1 歩以上に数えると強い相手に勝率が 51% → 37〜38% に下がったので採用していない
export const GUIDE_CURRENT: GuideParams = GUIDE_V2;

// 手番側から見た歩数の差（0 以上なら手番側が先着）
const lead = (s: GameState) => shortestDistance(s, (1 - s.turn) as Player) - shortestDistance(s, s.turn);

export function guideMove(s: GameState, params: GuideParams = GUIDE_V1): Move {
  if (params.lookahead) return lookaheadMove(s, params, params.lookahead);
  return simpleMove(s, params);
}

function lookaheadMove(s: GameState, params: GuideParams, level: number): Move {
  const me = s.turn;
  const op = (1 - me) as Player;
  // 相手がまだ遠いうちは壁を候補に入れない（温存）
  const walls = shortestDistance(s, op) <= params.saveWallsWhileOpponentFarther ? candidateWalls(s) : [];
  const candidates: Move[] = [...pawnMoves(s).map((p): Move => ({ type: 'pawn', x: p.x, y: p.y })), ...walls];
  let best = candidates[0];
  let bestScore = -Infinity;
  for (const m of candidates) {
    const after = applyMove(s, m);
    let score: number;
    if (after.winner === me) {
      score = Infinity;
    } else {
      // 相手は 1 段階浅い考え方で最善の手を返してくる、と想定する
      const reply = applyMove(after, guideMove(after, { ...params, lookahead: level - 1 }));
      if (reply.winner !== null) {
        score = -Infinity;
      } else {
        // 自分の手番に戻ったときの歩数の差 ＋ 残り壁の差。wallValue が 0 でも同点なら壁を温存する手を選ぶ
        score = lead(reply) + ((params.wallValue ?? 0) + 0.01) * (reply.wallsLeft[me] - reply.wallsLeft[1 - me]);
        if (params.threatValue) {
          score += params.threatValue * (maxDetour(reply, me) - maxDetour(reply, (1 - me) as Player));
        }
      }
    }
    if (score > bestScore) {
      bestScore = score;
      best = m;
    }
  }
  return best;
}

function simpleMove(s: GameState, params: GuideParams): Move {
  const me = s.turn;
  const op = (1 - me) as Player;
  const myDist = shortestDistance(s, me);
  const opDist = shortestDistance(s, op);
  const lead = opDist - myDist; // 0 以上なら手番の自分が先着

  const wantWall =
    s.wallsLeft[me] > 0 &&
    opDist <= params.saveWallsWhileOpponentFarther &&
    (lead < 0 || (lead <= params.defendLead && opDist <= params.holdUntil));

  if (wantWall) {
    let best: Move | null = null;
    let bestGain = -Infinity;
    for (const w of candidateWalls(s)) {
      const n = applyMove(s, w);
      const gain = shortestDistance(n, op) - opDist - (shortestDistance(n, me) - myDist);
      if (gain > bestGain) {
        bestGain = gain;
        best = w;
      }
    }
    if (best && bestGain >= params.minNetGain) return best;
  }

  const options = pawnMoves(s).map((p) => {
    const m: Move = { type: 'pawn', x: p.x, y: p.y };
    return { m, d: shortestDistance(applyMove(s, m), me), c: Math.abs(p.x - 4) };
  });
  options.sort((a, b) => a.d - b.d || a.c - b.c);
  return options[0].m;
}
