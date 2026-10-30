// 発見ログの局面から候補手を強制し、その後を探索 AI 同士で終局まで指し継ぐ。
// 評価値上の後悔だけでなく、候補手が実際の勝敗を変えるかを調べるために使う。

import { recentCells, search, type Weights } from '../engine/ai';
import {
  type GameState,
  type Player,
  applyMove,
  initialState,
  isLegal,
  isOver,
  moveFromString,
  moveToString,
} from '../engine/game';
import { type DiscoveryPosition, type Disagreement } from './discover';

export interface CounterfactualJob {
  disagreement: Disagreement;
  weights: Partial<Weights>;
  /** 方策列を終えた後、終局まで共通で使う読みの深さ。 */
  depth: number;
  /** 最初の候補手を含む policyPlies 手の間、guide側で使う深さ。 */
  guideDepth?: number;
  /** 最初の候補手を含む policyPlies 手の間、expert側で使う深さ。 */
  expertDepth?: number;
  /** 各分岐固有の深さを続ける手数。既定の1は従来の単独手比較。 */
  policyPlies?: number;
}

export interface BranchOutcome {
  winner: Player | null;
  remainingPlies: number;
  moves: string[];
}

export interface CounterfactualResult {
  disagreement: Disagreement;
  guide: BranchOutcome;
  expert: BranchOutcome;
}

export function replayPosition(position: DiscoveryPosition): GameState[] {
  const history = [initialState()];
  for (const text of position.moves.split(' ').filter(Boolean)) {
    const move = moveFromString(text);
    if (!move || !isLegal(history.at(-1)!, move)) throw new Error(`棋譜を再現できません: ${text}`);
    history.push(applyMove(history.at(-1)!, move));
  }
  const state = history.at(-1)!;
  if (state.ply !== position.ply || state.turn !== position.guidePlayer) {
    throw new Error(`局面が一致しません: ply ${state.ply}/${position.ply}, turn ${state.turn}/${position.guidePlayer}`);
  }
  return history;
}

function playBranch(
  position: DiscoveryPosition,
  firstMoveText: string,
  weights: Partial<Weights>,
  depth: number,
  policyDepth = depth,
  policyPlies = 1,
): BranchOutcome {
  const history = replayPosition(position);
  const firstMove = moveFromString(firstMoveText);
  if (!firstMove || !isLegal(history.at(-1)!, firstMove)) throw new Error(`候補手が合法ではありません: ${firstMoveText}`);

  const continuation: string[] = [firstMoveText];
  history.push(applyMove(history.at(-1)!, firstMove));
  while (!isOver(history.at(-1)!)) {
    const state = history.at(-1)!;
    const move = search(state, {
      maxDepth: continuation.length < policyPlies ? policyDepth : depth,
      weights,
      recent: recentCells(history, state.turn),
    }).move;
    continuation.push(moveToString(move));
    history.push(applyMove(state, move));
  }

  return {
    winner: history.at(-1)!.winner,
    remainingPlies: continuation.length,
    moves: continuation,
  };
}

export function evaluateCounterfactual(job: CounterfactualJob): CounterfactualResult {
  const policyPlies = job.policyPlies ?? 1;
  if (policyPlies < 1) throw new Error('policyPliesは1以上にしてください');
  return {
    disagreement: job.disagreement,
    guide: playBranch(
      job.disagreement.position,
      job.disagreement.guideMove,
      job.weights,
      job.depth,
      job.guideDepth,
      policyPlies,
    ),
    expert: playBranch(
      job.disagreement.position,
      job.disagreement.expertMove,
      job.weights,
      job.depth,
      job.expertDepth,
      policyPlies,
    ),
  };
}
