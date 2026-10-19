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
  // 相手の最短経路をふさぐ壁だけでなく、自分の経路を守る壁やコマ周りの壁も候補にする
  wideWalls?: boolean;
  // 先読みの評価で、壁を 1 枚使うことに課す歩数換算のコスト
  wallCost?: number;
  // 最後の壁を使い切るときだけ追加で課す歩数換算のコスト
  lastWallCost?: number;
  // 相手の壁 1 枚でこの歩数以上遠回りさせられるときだけ、脅威を減らす守りの壁を候補にする
  guardThreatAtLeast?: number;
}

export const GUIDE_V1: GuideParams = {
  minNetGain: 1,
  defendLead: -1,
  holdUntil: 0,
  saveWallsWhileOpponentFarther: 99,
};

export const GUIDE_V2: GuideParams = { ...GUIDE_V1, lookahead: 1 };
export const GUIDE_V3: GuideParams = { ...GUIDE_V1, lookahead: 2 };
// 第 4 版の候補だったもの（遠回りの差の半分）。第 4 世代には強かったが、全世代を相手にすると第 2 版より弱かった
export const GUIDE_V4: GuideParams = { ...GUIDE_V2, threatValue: 0.5 };

// docs/guide.md の第 4 版。全世代 × 2・3 手読みを相手にした平均勝率で選んだ
// - 基本: 第 2 版 + 返しのあとの「壁 1 枚で遠回りさせられる歩数」の差の 4 分の 1（平均 42.6%）
// - 上級: さらに「相手も基本の手順で考えて返してくる」と一段深く想像する（平均 49.0%）
// 壁の枚数を歩数に換算して足す版（wallValue）は、強い相手への勝率が下がったので採用していない
export const GUIDE_BASIC: GuideParams = { ...GUIDE_V2, threatValue: 0.25 };
export const GUIDE_ADVANCED: GuideParams = { ...GUIDE_V3, threatValue: 0.25 };
// 第 5 版: 第 7 世代との判断の食い違いを LLM で分類して作った候補を、全世代リーグで検証した。
// 壁を置きすぎる弱点を 0.5 歩のコストで抑え、4 歩以上の壁脅威があるときだけ守りの壁も読む。
// 第 1〜7 世代 × 2・3 手読みへの平均勝率は、第 4 版 47.8% → 第 5 版 55.4%。
export const GUIDE_V5: GuideParams = { ...GUIDE_ADVANCED, wallCost: 0.5, guardThreatAtLeast: 4 };
export const GUIDE_CURRENT: GuideParams = GUIDE_V5;

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
  const walls = shortestDistance(s, op) <= params.saveWallsWhileOpponentFarther ? guideWalls(s, params) : [];
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
        if (m.type === 'wall') {
          score -= params.wallCost ?? 0;
          if (s.wallsLeft[me] === 1) score -= params.lastWallCost ?? 0;
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

function guideWalls(s: GameState, params: GuideParams): Move[] {
  const base = candidateWalls(s, params.wideWalls ?? false);
  if (!params.guardThreatAtLeast || params.wideWalls) return base;

  const me = s.turn;
  const op = (1 - me) as Player;
  const threat = maxDetour(s, op);
  if (threat < params.guardThreatAtLeast) return base;
  const myDist = shortestDistance(s, me);
  const names = new Set(base.map((m) => JSON.stringify(m)));
  const guards = candidateWalls(s, true).filter((m) => {
    if (names.has(JSON.stringify(m))) return false;
    const after = applyMove(s, m);
    return shortestDistance(after, me) === myDist && maxDetour(after, op) < threat;
  });
  return [...base, ...guards];
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
    for (const w of guideWalls(s, params)) {
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
