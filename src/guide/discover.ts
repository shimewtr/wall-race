// 手引き AI と探索 AI の判断が食い違う局面を集める。
// LLM に渡せるよう、盤面・棋譜・局面の数値・探索上の損失を小さなレコードにまとめる。

import { maxDetour, recentCells, search, shortestRouteDiversity, type Weights } from '../engine/ai';
import {
  type GameState,
  type Move,
  type Player,
  WALL_SLOTS,
  applyMove,
  initialState,
  isOver,
  moveToString,
  shortestDistance,
} from '../engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../train/match';
import { GUIDE_CURRENT, type GuideParams, guideMove } from './policy';

export interface DiscoveryJob {
  weights: Partial<Weights>;
  depth: number;
  guidePlayer: Player;
  seed: number;
  params?: GuideParams;
}

export interface DiscoveryPosition {
  ply: number;
  guidePlayer: Player;
  pawns: [string, string];
  wallsLeft: [number, number];
  hWalls: string[];
  vWalls: string[];
  moves: string;
}

export interface Disagreement {
  position: DiscoveryPosition;
  guideMove: string;
  expertMove: string;
  guideType: Move['type'];
  expertType: Move['type'];
  expertScore: number;
  guideScore: number;
  regret: number;
  myDist: number;
  opDist: number;
  myWalls: number;
  opWalls: number;
  myThreat: number;
  opThreat: number;
  myRoutes: number;
  opRoutes: number;
  wallComparison?: {
    guide: WallEffects;
    expert: WallEffects;
  };
  pawnComparison?: {
    guide: PawnEffects;
    expert: PawnEffects;
  };
  wallPawnComparison?: {
    guide: WallEffects;
    expert: PawnEffects;
    guideAfterReply: ContinuationEffects;
    expertAfterReply: ContinuationEffects;
  };
  guideWon?: boolean;
}

export interface WallEffects {
  opDistanceGain: number;
  myDistanceCost: number;
  netGain: number;
  myThreat: number;
  opThreat: number;
  myRoutes: number;
  opRoutes: number;
}

export interface PawnEffects {
  progress: number;
  centerOffset: number;
  adjacent: boolean;
  backtrack: boolean;
  myThreat: number;
  opThreat: number;
  myRoutes: number;
  opRoutes: number;
}

export interface ContinuationEffects {
  reply: string | null;
  replyType: Move['type'] | null;
  winner: Player | null;
  myDist: number;
  opDist: number;
  raceLead: number;
  myWalls: number;
  opWalls: number;
  myThreat: number;
  opThreat: number;
}

export interface DiscoveryGame {
  guidePlayer: Player;
  winner: Player | null;
  moves: string[];
  disagreements: Disagreement[];
}

const sameMove = (a: Move, b: Move) =>
  a.type === b.type && a.x === b.x && a.y === b.y && (a.type === 'pawn' || (b.type === 'wall' && a.o === b.o));

const pawnName = (s: GameState, p: Player) => moveToString({ type: 'pawn', ...s.pawns[p] });

function wallNames(s: GameState, orientation: 'h' | 'v'): string[] {
  const walls = orientation === 'h' ? s.hWalls : s.vWalls;
  const result: string[] = [];
  for (let y = 0; y < WALL_SLOTS; y++) {
    for (let x = 0; x < WALL_SLOTS; x++) {
      if (walls[y * WALL_SLOTS + x]) result.push(moveToString({ type: 'wall', x, y, o: orientation }));
    }
  }
  return result;
}

const rounded = (n: number) => Math.round(n * 10) / 10;

function wallEffects(s: GameState, move: Move, me: Player, op: Player, myDist: number, opDist: number): WallEffects {
  const after = applyMove(s, move);
  const opDistanceGain = shortestDistance(after, op) - opDist;
  const myDistanceCost = shortestDistance(after, me) - myDist;
  return {
    opDistanceGain,
    myDistanceCost,
    netGain: opDistanceGain - myDistanceCost,
    myThreat: maxDetour(after, me),
    opThreat: maxDetour(after, op),
    myRoutes: rounded(shortestRouteDiversity(after, me)),
    opRoutes: rounded(shortestRouteDiversity(after, op)),
  };
}

function pawnEffects(
  s: GameState,
  move: Move,
  me: Player,
  op: Player,
  myDist: number,
  recent: ReturnType<typeof recentCells>,
): PawnEffects {
  const after = applyMove(s, move);
  return {
    progress: myDist - shortestDistance(after, me),
    centerOffset: Math.abs(move.x - 4),
    adjacent: Math.abs(move.x - after.pawns[op].x) + Math.abs(move.y - after.pawns[op].y) === 1,
    backtrack: recent.some((p) => p.x === move.x && p.y === move.y),
    myThreat: maxDetour(after, me),
    opThreat: maxDetour(after, op),
    myRoutes: rounded(shortestRouteDiversity(after, me)),
    opRoutes: rounded(shortestRouteDiversity(after, op)),
  };
}

function continuationEffects(
  afterCandidate: GameState,
  reply: Move | undefined,
  me: Player,
  op: Player,
): ContinuationEffects {
  const afterReply = reply ? applyMove(afterCandidate, reply) : afterCandidate;
  const myDist = shortestDistance(afterReply, me);
  const opDist = shortestDistance(afterReply, op);
  return {
    reply: reply ? moveToString(reply) : null,
    replyType: reply?.type ?? null,
    winner: afterReply.winner,
    myDist,
    opDist,
    raceLead: opDist - myDist,
    myWalls: afterReply.wallsLeft[me],
    opWalls: afterReply.wallsLeft[op],
    myThreat: maxDetour(afterReply, me),
    opThreat: maxDetour(afterReply, op),
  };
}

export function discoverGame(job: DiscoveryJob): DiscoveryGame {
  const params = job.params ?? GUIDE_CURRENT;
  const rng = rngFrom(job.seed);
  const history: GameState[] = [initialState()];
  const moves: string[] = [];
  const disagreements: Disagreement[] = [];
  let s = history[0];

  while (!isOver(s)) {
    const currentDepth = depthAt(s.ply, job.depth, OPENING_PLIES);
    let move: Move;
    if (s.turn === job.guidePlayer) {
      const guide = guideMove(s, params, recentCells(history, s.turn));
      const expert = search(s, {
        maxDepth: currentDepth,
        weights: job.weights,
        recent: recentCells(history, s.turn),
      });
      move = guide;
      if (!sameMove(guide, expert.move)) {
        const me = s.turn;
        const op = (1 - me) as Player;
        const recent = recentCells(history, me);
        const myDist = shortestDistance(s, me);
        const opDist = shortestDistance(s, op);
        const after = applyMove(s, guide);
        // 根の探索と同じ読みの深さになるよう、手引きの手の後を 1 段浅く読む。
        // 手引きの手で即勝ちした場合は損失 0 として、発見候補から自然に外す。
        const guideReply = isOver(after)
          ? undefined
          : search(after, {
              maxDepth: Math.max(1, currentDepth - 1),
              weights: job.weights,
              recent: recentCells([...history, after], after.turn),
            });
        const guideScore = guideReply ? -guideReply.score : expert.score;
        const wallPawnComparison =
          guide.type === 'wall' && expert.move.type === 'pawn'
            ? (() => {
                const afterExpert = applyMove(s, expert.move);
                const expertReply = isOver(afterExpert)
                  ? undefined
                  : search(afterExpert, {
                      maxDepth: Math.max(1, currentDepth - 1),
                      weights: job.weights,
                      recent: recentCells([...history, afterExpert], afterExpert.turn),
                    });
                return {
                  guide: wallEffects(s, guide, me, op, myDist, opDist),
                  expert: pawnEffects(s, expert.move, me, op, myDist, recent),
                  guideAfterReply: continuationEffects(after, guideReply?.move, me, op),
                  expertAfterReply: continuationEffects(afterExpert, expertReply?.move, me, op),
                };
              })()
            : undefined;
        disagreements.push({
          position: {
            ply: s.ply,
            guidePlayer: job.guidePlayer,
            pawns: [pawnName(s, 0), pawnName(s, 1)],
            wallsLeft: [s.wallsLeft[0], s.wallsLeft[1]],
            hWalls: wallNames(s, 'h'),
            vWalls: wallNames(s, 'v'),
            moves: moves.join(' '),
          },
          guideMove: moveToString(guide),
          expertMove: moveToString(expert.move),
          guideType: guide.type,
          expertType: expert.move.type,
          expertScore: rounded(expert.score),
          guideScore: rounded(guideScore),
          regret: rounded(expert.score - guideScore),
          myDist,
          opDist,
          myWalls: s.wallsLeft[me],
          opWalls: s.wallsLeft[op],
          myThreat: maxDetour(s, me),
          opThreat: maxDetour(s, op),
          myRoutes: rounded(shortestRouteDiversity(s, me)),
          opRoutes: rounded(shortestRouteDiversity(s, op)),
          wallComparison:
            guide.type === 'wall' && expert.move.type === 'wall'
              ? {
                  guide: wallEffects(s, guide, me, op, myDist, opDist),
                  expert: wallEffects(s, expert.move, me, op, myDist, opDist),
                }
              : undefined,
          pawnComparison:
            guide.type === 'pawn' && expert.move.type === 'pawn'
              ? {
                  guide: pawnEffects(s, guide, me, op, myDist, recent),
                  expert: pawnEffects(s, expert.move, me, op, myDist, recent),
                }
              : undefined,
          wallPawnComparison,
        });
      }
    } else {
      move = search(s, {
        maxDepth: currentDepth,
        weights: job.weights,
        margin: 1,
        pickFromTop: pickAt(s.ply, OPENING_PLIES),
        rng,
        recent: recentCells(history, s.turn),
      }).move;
    }
    moves.push(moveToString(move));
    s = applyMove(s, move);
    history.push(s);
  }

  for (const d of disagreements) d.guideWon = s.winner === job.guidePlayer;
  return { guidePlayer: job.guidePlayer, winner: s.winner, moves, disagreements };
}
