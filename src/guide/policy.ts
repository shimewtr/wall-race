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

import { candidateWalls } from '../engine/ai';
import { type GameState, type Move, type Player, applyMove, pawnMoves, shortestDistance } from '../engine/game';

export interface GuideParams {
  // 壁を置くのは「相手の増加 − 自分の増加」がこの歩数以上のときだけ
  minNetGain: number;
  // 自分がリードしていても、リードがこの歩数以下で相手がゴール間近（holdUntil 歩以内）なら壁で突き放す
  defendLead: number;
  holdUntil: number;
  // 相手の残り歩数がこれより多いうちは、負けていても壁を温存する（序盤に使い切らない）
  saveWallsWhileOpponentFarther: number;
}

export const GUIDE_V1: GuideParams = {
  minNetGain: 1,
  defendLead: -1,
  holdUntil: 0,
  saveWallsWhileOpponentFarther: 99,
};

export function guideMove(s: GameState, params: GuideParams = GUIDE_V1): Move {
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
