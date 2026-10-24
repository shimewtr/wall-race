// ルール上等価な対称局面で、探索 AI の合法手・評価値・選択手が一致するか調べる。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/symmetry.ts --games 20 --positions 120 --depth 3 --sample-every 4

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { checkSymmetry, type Symmetry, type SymmetryCheck } from '../../src/analysis/symmetry';
import { FEATURE_NAMES, recentCells, search, type Feature } from '../../src/engine/ai';
import { type GameState, applyMove, initialState, isOver, moveToString } from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '20' },
    positions: { type: 'string', default: '120' },
    depth: { type: 'string', default: '3' },
    'sample-every': { type: 'string', default: '4' },
    examples: { type: 'string', default: '10' },
    disable: { type: 'string', default: '' },
  },
});

const champion = JSON.parse(
  readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'),
);
const games = Number(args.games);
const maxPositions = Number(args.positions);
const depth = Number(args.depth);
const sampleEvery = Number(args['sample-every']);
const maxExamples = Number(args.examples);
const symmetries: Symmetry[] = ['left-right', 'swap-vertical'];
const disabled = args.disable
  .split(',')
  .map((name) => name.trim())
  .filter(Boolean);
for (const name of disabled) {
  if (!FEATURE_NAMES.includes(name as Feature)) throw new Error(`unknown feature: ${name}`);
}
const weights = { ...champion.weights };
for (const name of disabled) weights[name as Feature] = 0;

interface Observation extends SymmetryCheck {
  game: number;
  ply: number;
  turn: number;
  position: string;
}

const observations: Observation[] = [];
let sampled = 0;

for (let game = 0; game < games && sampled < maxPositions; game++) {
  const rng = rngFrom(71_000_000 + game);
  const history: GameState[] = [initialState()];
  let state = history[0];
  const moves: string[] = [];

  while (!isOver(state) && sampled < maxPositions) {
    if (state.ply % sampleEvery === game % sampleEvery) {
      // 履歴由来の repeatPenalty は別の検査対象にし、ここでは盤面だけの対称性を測る。
      const options = { maxDepth: depth, weights, repeatPenalty: 0 };
      const original = search(state, options);
      const position = moves.join(' ');
      for (const symmetry of symmetries) {
        observations.push({
          ...checkSymmetry(state, options, symmetry, original),
          game: game + 1,
          ply: state.ply,
          turn: state.turn,
          position,
        });
      }
      sampled++;
    }

    const move = search(state, {
      maxDepth: depthAt(state.ply, depth, OPENING_PLIES),
      weights,
      margin: 1,
      pickFromTop: pickAt(state.ply, OPENING_PLIES),
      rng,
      recent: recentCells(history, state.turn),
    }).move;
    moves.push(moveToString(move));
    state = applyMove(state, move);
    history.push(state);
  }
}

const pct = (count: number, total: number) =>
  total === 0 ? '-' : `${((count / total) * 100).toFixed(1)}%`;
const average = (values: number[]) =>
  values.length === 0 ? '-' : (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2);

console.log(`# 第 ${champion.generation} 世代の対称性分析`);
console.log();
console.log(`収集: ${Math.min(games, new Set(observations.map((row) => row.game)).size)} 局から ${sampled} 局面`);
console.log(`探索: ${depth} 手読み、${sampleEvery} 手間隔、履歴ペナルティなし`);
console.log(`無効化した特徴量: ${disabled.join(', ') || 'なし'}`);
console.log('左右反転と、上下反転＋プレイヤー交換を検査しています。');

for (const symmetry of symmetries) {
  const rows = observations.filter((row) => row.symmetry === symmetry);
  const legalMismatches = rows.filter((row) => !row.legalMovesMatch);
  const scoreMismatches = rows.filter((row) => Math.abs(row.scoreDelta) > 1e-9);
  const moveMismatches = rows.filter((row) => !row.moveMatches);
  const name = symmetry === 'left-right' ? '左右反転' : '上下反転＋プレイヤー交換';
  console.log(`\n## ${name}`);
  console.log(`- 合法手集合の不一致: ${legalMismatches.length} / ${rows.length}（${pct(legalMismatches.length, rows.length)}）`);
  console.log(`- 探索評価値の不一致: ${scoreMismatches.length} / ${rows.length}（${pct(scoreMismatches.length, rows.length)}）`);
  console.log(`- 選択手の不一致: ${moveMismatches.length} / ${rows.length}（${pct(moveMismatches.length, rows.length)}）`);
  console.log(`- 探索ノード比（変換後 / 元）平均: ${average(rows.map((row) => row.transformedNodes / row.originalNodes))}`);

  const examples = rows
    .filter((row) => !row.legalMovesMatch || Math.abs(row.scoreDelta) > 1e-9 || !row.moveMatches)
    .slice(0, maxExamples);
  if (examples.length) {
    console.log('- 不一致例:');
    for (const row of examples) {
      console.log(
        `  - 対局${row.game}・${row.ply}手目・手番${row.turn}: 手 ${row.expectedMove} → ${row.actualMove} / 評価 ${row.originalScore.toFixed(3)} → ${row.transformedScore.toFixed(3)} / 合法手 ${row.legalMovesMatch ? '一致' : '不一致'}`,
      );
      console.log(`    棋譜: ${row.position || '（初期局面）'}`);
    }
  }
}

console.log('\n## 判定の読み方');
console.log('- 合法手集合の不一致はルール変換またはルールエンジンの問題です。');
console.log('- 評価値が同じで選択手だけ違う場合は、同点処理や候補順への依存が疑われます。');
console.log('- 評価値も違う場合は、特徴量・探索・キャッシュの非対称性を優先して調べます。');
