// 学習中の重み変化が、3手読みの判断を第12世代の4手読みへ近づけているか調べる。
// 重い自己対戦・探索を含むためGitHub ActionsのExperimentで実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { FEATURE_NAMES, type Weights, recentCells, search, withDefaults } from '../../src/engine/ai';
import { type GameState, applyMove, initialState, isOver, moveToString } from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '80' },
    positions: { type: 'string', default: '600' },
    'sample-every': { type: 'string', default: '4' },
    fractions: { type: 'string', default: '0,0.25,0.5,0.75,1' },
    seed: { type: 'string', default: '106000000' },
    examples: { type: 'string', default: '12' },
  },
});

const root = join(import.meta.dirname, '..', '..');
const championFile = JSON.parse(readFileSync(join(root, 'training', 'champion.json'), 'utf8')) as {
  generation: number;
  weights: Partial<Weights>;
};
const stateFile = JSON.parse(readFileSync(join(root, 'training', 'state.json'), 'utf8')) as {
  theta: Partial<Weights>;
  stepIteration?: number;
};
const champion = withDefaults(championFile.weights);
const current = withDefaults(stateFile.theta);
const games = Number(args.games);
const maxPositions = Number(args.positions);
const sampleEvery = Number(args['sample-every']);
const seed = Number(args.seed);
const maxExamples = Number(args.examples);
const fractions = args.fractions.split(',').map(Number);
if (fractions.some((fraction) => !Number.isFinite(fraction))) throw new Error('fractions must be comma-separated numbers');

const interpolate = (fraction: number): Weights => Object.fromEntries(
  FEATURE_NAMES.map((feature) => [feature, champion[feature] + fraction * (current[feature] - champion[feature])]),
) as Weights;
const students = fractions.map((fraction) => ({ fraction, weights: interpolate(fraction) }));

interface Sample {
  game: number;
  ply: number;
  phase: '序盤' | '中盤' | '終盤';
  position: string;
  state: GameState;
}
const phaseAt = (ply: number): Sample['phase'] => ply < 20 ? '序盤' : ply < 40 ? '中盤' : '終盤';
const samples: Sample[] = [];
for (let game = 0; game < games && samples.length < maxPositions; game++) {
  const rng = rngFrom(seed + game);
  const history: GameState[] = [initialState()];
  const moves: string[] = [];
  let state = history[0];
  while (!isOver(state) && samples.length < maxPositions) {
    if (state.ply % sampleEvery === game % sampleEvery) {
      samples.push({ game: game + 1, ply: state.ply, phase: phaseAt(state.ply), position: moves.join(' '), state });
    }
    const move = search(state, {
      maxDepth: depthAt(state.ply, 3, OPENING_PLIES),
      weights: champion,
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

interface Observation {
  game: number;
  ply: number;
  phase: Sample['phase'];
  position: string;
  champion3: string;
  teacher4: string;
  choices: Array<{ fraction: number; move: string }>;
}
const observations: Observation[] = samples.map((sample) => {
  const champion3 = moveToString(search(sample.state, { maxDepth: 3, weights: champion, repeatPenalty: 0 }).move);
  const teacher4 = moveToString(search(sample.state, { maxDepth: 4, weights: champion, repeatPenalty: 0 }).move);
  return {
    game: sample.game,
    ply: sample.ply,
    phase: sample.phase,
    position: sample.position,
    champion3,
    teacher4,
    choices: students.map(({ fraction, weights }) => ({
      fraction,
      move: moveToString(search(sample.state, { maxDepth: 3, weights, repeatPenalty: 0 }).move),
    })),
  };
});

const critical = observations.filter((row) => row.champion3 !== row.teacher4);
const pct = (count: number, total: number) => total ? `${(100 * count / total).toFixed(1)}%` : '-';
const results = students.map(({ fraction }) => {
  const moveAt = (row: Observation) => row.choices.find((choice) => choice.fraction === fraction)!.move;
  const teacherMatches = observations.filter((row) => moveAt(row) === row.teacher4).length;
  const championChanges = observations.filter((row) => moveAt(row) !== row.champion3).length;
  const criticalTeacherMatches = critical.filter((row) => moveAt(row) === row.teacher4).length;
  const criticalChampionMatches = critical.filter((row) => moveAt(row) === row.champion3).length;
  return {
    fraction,
    teacherMatches,
    teacherRate: teacherMatches / observations.length,
    championChanges,
    championChangeRate: championChanges / observations.length,
    criticalTeacherMatches,
    criticalTeacherRate: critical.length ? criticalTeacherMatches / critical.length : 0,
    criticalChampionMatches,
    criticalOther: critical.length - criticalTeacherMatches - criticalChampionMatches,
  };
});

console.log(`# 第${championFile.generation}世代の4手読み教師一致分析`);
console.log(`収集: ${new Set(observations.map((row) => row.game)).size}局から${observations.length}局面、seed=${seed}`);
console.log(`3手と4手の判断差: ${critical.length}/${observations.length}（${pct(critical.length, observations.length)}）`);
console.log(`学習中stepIteration=${stateFile.stepIteration ?? 0}`);
console.log('\n| チャンピオン→学習中 | 4手教師との一致 | 3手チャンピオンから変更 | 判断差局面で教師へ変更 | 判断差局面で元のまま | それ以外 |');
console.log('| ---: | ---: | ---: | ---: | ---: | ---: |');
for (const row of results) {
  console.log(`| ${(row.fraction * 100).toFixed(0)}% | ${row.teacherMatches}/${observations.length} (${pct(row.teacherMatches, observations.length)}) | ${row.championChanges}/${observations.length} (${pct(row.championChanges, observations.length)}) | ${row.criticalTeacherMatches}/${critical.length} (${pct(row.criticalTeacherMatches, critical.length)}) | ${row.criticalChampionMatches}/${critical.length} | ${row.criticalOther}/${critical.length} |`);
}

console.log('\n## フェーズ別・学習中重みの教師一致');
const currentFraction = fractions.at(-1)!;
for (const phase of ['序盤', '中盤', '終盤'] as const) {
  const rows = observations.filter((row) => row.phase === phase);
  const matches = rows.filter((row) => row.choices.find((choice) => choice.fraction === currentFraction)!.move === row.teacher4).length;
  const base = rows.filter((row) => row.champion3 === row.teacher4).length;
  console.log(`- ${phase}: チャンピオン3手 ${pct(base, rows.length)} → 学習中3手 ${pct(matches, rows.length)}（${matches - base >= 0 ? '+' : ''}${matches - base}局面）`);
}

console.log('\n## 学習中重みが教師側へ直した例');
const fixed = critical.filter((row) => row.choices.find((choice) => choice.fraction === currentFraction)!.move === row.teacher4);
if (!fixed.length) console.log('- なし');
for (const row of fixed.slice(0, maxExamples)) {
  console.log(`- 対局${row.game}・${row.ply}手目（${row.phase}）: 3手 ${row.champion3} → 教師/学習中 ${row.teacher4} / 棋譜 ${row.position || '（初期局面）'}`);
}

console.log('\n## 判定');
const baseline = results.find((row) => row.fraction === 0);
const endpoint = results.find((row) => row.fraction === currentFraction)!;
if (baseline && endpoint.teacherRate > baseline.teacherRate && endpoint.criticalTeacherMatches > endpoint.criticalOther) {
  console.log('- 学習中の方向は4手読み教師への一致を増やしている。独立seedで再現後、教師一致を目的にした短期学習を試す。');
} else {
  console.log('- 学習中の方向は4手読み教師へ安定して近づいていない。蒸留学習より、3手読みで表現できない新特徴量を優先する。');
}

writeFileSync('experiment-output.json', `${JSON.stringify({
  generation: championFile.generation,
  source: { games, maxPositions, sampleEvery, seed, fractions, stepIteration: stateFile.stepIteration ?? 0 },
  critical: critical.length,
  results,
  fixedExamples: fixed.slice(0, maxExamples),
}, null, 2)}\n`);
