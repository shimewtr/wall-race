// 異なる読みの深さを直接対戦させ、深い側が勝つ対局と負ける対局の手順差を調べる。
// 重い処理なのでGitHub ActionsのExperimentで実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import type { Player } from '../../src/engine/game';
import type { GameRecord, MoveRecord, RecordJob } from '../../src/train/record';
import type { Weights } from '../../src/engine/ai';
import { RecordPool } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    pairs: { type: 'string', default: '200' },
    shallow: { type: 'string', default: '4' },
    deep: { type: 'string', default: '5' },
    seed: { type: 'string', default: '135000000' },
  },
});
const pairs = Number(args.pairs);
const shallow = Number(args.shallow);
const deep = Number(args.deep);
const seed = Number(args.seed);
if (deep <= shallow) throw new Error('deepはshallowより大きくしてください');

const root = join(import.meta.dirname, '..', '..');
const champion = JSON.parse(readFileSync(join(root, 'training', 'champion.json'), 'utf8')) as {
  generation: number;
  weights: Weights;
};
interface TaggedRecord {
  record: GameRecord;
  depths: [number, number];
  deepPlayer: Player;
}
const jobs: Array<RecordJob & { depths: [number, number] }> = [];
for (let pair = 0; pair < pairs; pair++) {
  jobs.push({ weights: champion.weights, depth: shallow, depths: [deep, shallow], seed: seed + pair });
  jobs.push({ weights: champion.weights, depth: shallow, depths: [shallow, deep], seed: seed + pair });
}
const pool = new RecordPool();
const raw = await pool.runAll(jobs);
await pool.close();
const records: TaggedRecord[] = raw.map((record, index) => ({
  record,
  depths: jobs[index].depths,
  deepPlayer: jobs[index].depths[0] === deep ? 0 : 1,
}));

const pct = (n: number, d: number) => d ? `${(n / d * 100).toFixed(1)}%` : '-';
const avg = (values: number[]) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
interface TaggedMove extends MoveRecord {
  depth: number;
  result: 'win' | 'loss' | 'draw';
}
const moves: TaggedMove[] = records.flatMap(({ record, depths, deepPlayer }) => record.moves.map((move) => ({
  ...move,
  depth: depths[move.player],
  result: record.winner === null ? 'draw' : record.winner === deepPlayer ? 'win' : 'loss',
})));
const decided = records.filter(({ record }) => record.winner !== null);
const deepWins = decided.filter(({ record, deepPlayer }) => record.winner === deepPlayer).length;

interface Summary {
  moves: number;
  wallRate: number;
  forwardRate: number;
  equalRate: number;
  detourRate: number;
  selfHarmWallRate: number;
  wallGain: number;
  firstWallPly: number;
}
const summarize = (depth: number, result?: 'win' | 'loss'): Summary => {
  const selected = moves.filter((move) => move.depth === depth && (!result || move.result === result));
  const pawns = selected.filter((move) => move.type === 'pawn');
  const walls = selected.filter((move) => move.type === 'wall');
  const firstWalls = records.flatMap(({ record, depths, deepPlayer }) => {
    const gameResult = record.winner === null ? 'draw' : record.winner === deepPlayer ? 'win' : 'loss';
    if (result && gameResult !== result) return [];
    return ([0, 1] as Player[]).filter((player) => depths[player] === depth).flatMap((player) => {
      const first = record.moves.find((move) => move.player === player && move.type === 'wall');
      return first ? [first.ply] : [];
    });
  });
  return {
    moves: selected.length,
    wallRate: walls.length / selected.length,
    forwardRate: pawns.filter((move) => move.myDistAfter < move.myDist).length / pawns.length,
    equalRate: pawns.filter((move) => move.myDistAfter === move.myDist).length / pawns.length,
    detourRate: pawns.filter((move) => move.myDistAfter > move.myDist).length / pawns.length,
    selfHarmWallRate: walls.filter((move) => move.myDistAfter > move.myDist).length / walls.length,
    wallGain: avg(walls.map((move) => move.opDistAfter - move.opDist)),
    firstWallPly: avg(firstWalls),
  };
};
const summaries = {
  deep: summarize(deep),
  shallow: summarize(shallow),
  deepWhenWin: summarize(deep, 'win'),
  deepWhenLoss: summarize(deep, 'loss'),
  shallowWhenDeepWins: summarize(shallow, 'win'),
  shallowWhenDeepLoses: summarize(shallow, 'loss'),
};

const printSummary = (name: string, value: Summary) => {
  console.log(`- ${name}: 壁${pct(value.wallRate, 1)}、前進${pct(value.forwardRate, 1)}、同距離${pct(value.equalRate, 1)}、遠回り${pct(value.detourRate, 1)}、自損壁${pct(value.selfHarmWallRate, 1)}、壁効果${value.wallGain.toFixed(2)}歩、初壁${value.firstWallPly.toFixed(1)}手目（${value.moves}手）`);
};

console.log(`# 第${champion.generation}世代 ${deep}手対${shallow}手の直接対戦分析`);
console.log(`対戦: ${pairs}組・${records.length}局、${deep}手側勝率${pct(deepWins, decided.length)}、引き分け${records.length - decided.length}、seed=${seed}`);
console.log('\n## 深さ別の全体傾向');
printSummary(`${deep}手`, summaries.deep);
printSummary(`${shallow}手`, summaries.shallow);
console.log(`\n## ${deep}手側の勝敗別`);
printSummary(`${deep}手・勝ち`, summaries.deepWhenWin);
printSummary(`${deep}手・負け`, summaries.deepWhenLoss);
printSummary(`${shallow}手・相手が勝ち`, summaries.shallowWhenDeepWins);
printSummary(`${shallow}手・相手が負け`, summaries.shallowWhenDeepLoses);

console.log('\n## 形勢別の壁率（相手残り歩数−自分残り歩数）');
for (let lead = -3; lead <= 3; lead++) {
  const label = lead === -3 ? '-3以下' : lead === 3 ? '+3以上' : `${lead >= 0 ? '+' : ''}${lead}`;
  const inBucket = (move: TaggedMove) => Math.max(-3, Math.min(3, move.opDist - move.myDist)) === lead;
  const cells = [deep, shallow].map((depth) => {
    const selected = moves.filter((move) => move.depth === depth && move.myWalls > 0 && inBucket(move));
    return `${depth}手 ${pct(selected.filter((move) => move.type === 'wall').length, selected.length)} (${selected.length})`;
  });
  console.log(`- ${label}: ${cells.join(' / ')}`);
}

writeFileSync('experiment-output.json', `${JSON.stringify({
  generation: champion.generation,
  source: { pairs, shallow, deep, seed },
  totals: { games: records.length, decided: decided.length, deepWins, deepWinRate: deepWins / decided.length },
  summaries,
}, null, 2)}\n`);
