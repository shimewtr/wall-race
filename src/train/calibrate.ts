// 探索の評価値を「勝率」に換算するための式を、自己対戦の結果から求める。
// 評価値は重みの組み合わせなので「何歩分」とは読めない。そこで
//   P(手番側が勝つ) = 1 / (1 + exp(-(a × 評価値 + b)))
// のロジスティック回帰で、実際の勝ち負けに合う a, b を求める。b は読みの深さの偶奇などによる手番側の片寄りを吸収する。

import { WIN } from '../engine/ai';
import type { GameRecord } from './record';

export interface Calibration {
  generation: number;
  a: number;
  b: number;
  // 当てはめに使った局面数
  samples: number;
}

export const winRate = (score: number, cal: Pick<Calibration, 'a' | 'b'>): number => {
  if (score > WIN / 2) return 1;
  if (score < -WIN / 2) return 0;
  return 1 / (1 + Math.exp(-(cal.a * score + cal.b)));
};

// 自己対戦の記録から (評価値, 手番側が勝ったか) を集めてロジスティック回帰する（ニュートン法）
export function fitCalibration(records: GameRecord[], generation: number): Calibration {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const g of records) {
    if (g.winner === null) continue;
    for (const m of g.moves) {
      if (Math.abs(m.score) > WIN / 2) continue; // 勝ちを読み切った局面は換算しなくてよい
      xs.push(m.score);
      ys.push(g.winner === m.player ? 1 : 0);
    }
  }
  // 評価値は数百点の幅があるので、尺度をそろえてから解く
  const scale = Math.max(1, Math.sqrt(xs.reduce((s, x) => s + x * x, 0) / Math.max(1, xs.length)));
  let a = 0;
  let b = 0;
  for (let iter = 0; iter < 50; iter++) {
    let ga = 0;
    let gb = 0;
    let haa = 0;
    let hab = 0;
    let hbb = 0;
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i] / scale;
      const p = 1 / (1 + Math.exp(-(a * x + b)));
      const w = p * (1 - p);
      ga += (ys[i] - p) * x;
      gb += ys[i] - p;
      haa += w * x * x;
      hab += w * x;
      hbb += w;
    }
    const det = haa * hbb - hab * hab;
    if (det <= 1e-12) break;
    const da = (hbb * ga - hab * gb) / det;
    const db = (haa * gb - hab * ga) / det;
    a += da;
    b += db;
    if (Math.abs(da) + Math.abs(db) < 1e-9) break;
  }
  return { generation, a: a / scale, b, samples: xs.length };
}
