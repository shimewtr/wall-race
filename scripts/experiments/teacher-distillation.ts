// 深い読みの選択を教師ラベルにし、浅い読み用の評価重みを固定局面上で蒸留する。
// 学習用と検証用の局面を分離し、候補は最後に通常対局400局でチャンピオンと比較する。
// 重い処理なのでGitHub ActionsのExperimentで実行する。

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { FEATURE_NAMES, recentCells, search, type Feature, type Weights, withDefaults } from '../../src/engine/ai';
import { type GameState, applyMove, initialState, isOver, moveToString } from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';
import { PairPool, type PairJob } from '../lib/pool';

const { values: args } = parseArgs({
  options: {
    positions: { type: 'string', default: '120' },
    passes: { type: 'string', default: '2' },
    pairs: { type: 'string', default: '200' },
    'student-depth': { type: 'string', default: '4' },
    'teacher-depth': { type: 'string', default: '5' },
    seed: { type: 'string', default: '139000000' },
  },
});
const positionCount = Number(args.positions);
const passes = Number(args.passes);
const pairs = Number(args.pairs);
const studentDepth = Number(args['student-depth']);
const teacherDepth = Number(args['teacher-depth']);
const seed = Number(args.seed);
if (teacherDepth <= studentDepth) throw new Error('teacher-depthはstudent-depthより大きくしてください');

const root = join(import.meta.dirname, '..', '..');
const championFile = JSON.parse(readFileSync(join(root, 'training', 'champion.json'), 'utf8')) as {
  generation: number;
  weights: Partial<Weights>;
};
const champion = withDefaults(championFile.weights);
const tuned = FEATURE_NAMES.filter((feature): feature is Feature => feature !== 'pathDiff');
const steps: Record<Feature, number> = {
  pathDiff: 0,
  rowDiff: 1,
  wallDiff: 1,
  urgency: 4,
  wallsWhenClose: 2,
  center: 1,
  adjacent: 2,
  lockedRace: 5,
  wallThreat: 2,
  netWallThreat: 2,
  wallChain: 1,
  routeDiversity: 2,
};

interface LabelledPosition {
  state: GameState;
  teacherMove: string;
  baselineMove: string;
  critical: boolean;
}
const collect = (baseSeed: number): GameState[] => {
  const states: GameState[] = [];
  for (let game = 0; states.length < positionCount; game++) {
    const rng = rngFrom(baseSeed + game);
    const history: GameState[] = [initialState()];
    let state = history[0];
    while (!isOver(state) && states.length < positionCount) {
      if (state.ply >= OPENING_PLIES && state.ply % 4 === game % 4) states.push(state);
      const move = search(state, {
        maxDepth: depthAt(state.ply, studentDepth, OPENING_PLIES),
        weights: champion,
        margin: 1,
        pickFromTop: pickAt(state.ply, OPENING_PLIES),
        rng,
        recent: recentCells(history, state.turn),
      }).move;
      state = applyMove(state, move);
      history.push(state);
    }
  }
  return states;
};
const label = (states: GameState[]): LabelledPosition[] => states.map((state) => {
  const baselineMove = moveToString(search(state, {
    maxDepth: studentDepth,
    weights: champion,
    repeatPenalty: 0,
  }).move);
  const teacherMove = moveToString(search(state, {
    maxDepth: teacherDepth,
    weights: champion,
    repeatPenalty: 0,
  }).move);
  return { state, teacherMove, baselineMove, critical: teacherMove !== baselineMove };
});
const training = label(collect(seed));
const validation = label(collect(seed + 1_000_000));

interface Agreement {
  score: number;
  rate: number;
  criticalRate: number;
  stableRate: number;
  matches: number;
  criticalMatches: number;
  critical: number;
  stableBreaks: number;
}
const agreement = (weights: Weights, rows: LabelledPosition[]): Agreement => {
  let matches = 0;
  let criticalMatches = 0;
  let stableMatches = 0;
  const critical = rows.filter((row) => row.critical).length;
  const stable = rows.length - critical;
  for (const row of rows) {
    const move = moveToString(search(row.state, {
      maxDepth: studentDepth,
      weights,
      repeatPenalty: 0,
    }).move);
    if (move === row.teacherMove) {
      matches++;
      if (row.critical) criticalMatches++;
      else stableMatches++;
    }
  }
  const criticalRate = critical ? criticalMatches / critical : 0;
  const stableRate = stable ? stableMatches / stable : 0;
  return {
    score: (criticalRate + stableRate) / 2,
    rate: matches / rows.length,
    criticalRate,
    stableRate,
    matches,
    criticalMatches,
    critical,
    stableBreaks: stable - stableMatches,
  };
};

interface StepResult {
  pass: number;
  feature: Feature;
  direction: -1 | 0 | 1;
  value: number;
  training: Agreement;
}
const history: StepResult[] = [];
const snapshots: Array<{ name: string; weights: Weights; training: Agreement; validation: Agreement }> = [];
let weights = { ...champion };
let current = agreement(weights, training);
snapshots.push({ name: 'baseline', weights: { ...weights }, training: current, validation: agreement(weights, validation) });
for (let pass = 0; pass < passes; pass++) {
  // seedごとに順序を変え、特定の特徴順への依存を独立実験で検出できるようにする。
  const rng = rngFrom(seed + 2_000_000 + pass);
  const order = [...tuned];
  for (let index = order.length - 1; index > 0; index--) {
    const swap = Math.floor(rng() * (index + 1));
    [order[index], order[swap]] = [order[swap], order[index]];
  }
  for (const feature of order) {
    const candidates = ([-1, 1] as const).map((direction) => {
      const candidate = { ...weights, [feature]: weights[feature] + direction * steps[feature] };
      return { direction, weights: candidate, result: agreement(candidate, training) };
    });
    const best = candidates.sort((a, b) => b.result.score - a.result.score || b.result.rate - a.result.rate)[0];
    let direction: -1 | 0 | 1 = 0;
    if (best.result.score > current.score || (best.result.score === current.score && best.result.rate > current.rate)) {
      weights = best.weights;
      current = best.result;
      direction = best.direction;
    }
    history.push({ pass: pass + 1, feature, direction, value: weights[feature], training: current });
  }
  snapshots.push({
    name: `pass${pass + 1}`,
    weights: { ...weights },
    training: current,
    validation: agreement(weights, validation),
  });
}
const selected = [...snapshots].sort((a, b) =>
  b.validation.score - a.validation.score || b.validation.rate - a.validation.rate
)[0];

const pool = new PairPool();
const jobs: PairJob[] = Array.from({ length: pairs }, (_, index) => ({
  a: selected.weights,
  b: champion,
  opts: {
    depth: studentDepth,
    margin: 1,
    openingPlies: OPENING_PLIES,
    seed: seed + 10_000_000 + index,
  },
}));
const scores = await pool.runAll(jobs);
await pool.close();
const matchRate = scores.reduce((sum, value) => sum + value, 0) / (pairs * 2);
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const describe = (value: Agreement) =>
  `全体${pct(value.rate)}、判断差${value.criticalMatches}/${value.critical} (${pct(value.criticalRate)})、安定破壊${value.stableBreaks}、均衡${pct(value.score)}`;

console.log(`# 第${championFile.generation}世代 ${teacherDepth}手→${studentDepth}手の教師蒸留`);
console.log(`学習${training.length}局面（判断差${training.filter((row) => row.critical).length}）、検証${validation.length}局面（判断差${validation.filter((row) => row.critical).length}）、seed=${seed}`);
console.log('\n## 局面一致');
for (const snapshot of snapshots) {
  console.log(`- ${snapshot.name}: 学習 ${describe(snapshot.training)} / 検証 ${describe(snapshot.validation)}`);
}
console.log(`- 採用候補: ${selected.name}`);
console.log('\n## 変更した重み');
for (const feature of tuned) {
  if (selected.weights[feature] !== champion[feature]) {
    console.log(`- ${feature}: ${champion[feature].toFixed(3)} → ${selected.weights[feature].toFixed(3)}`);
  }
}
console.log(`\n## 通常対局\n- 候補 対 チャンピオン（${studentDepth}手読み）: ${pct(matchRate)}（${pairs * 2}局）`);

writeFileSync('experiment-output.json', `${JSON.stringify({
  generation: championFile.generation,
  source: { positionCount, passes, pairs, studentDepth, teacherDepth, seed },
  snapshots,
  selected,
  history,
  matchRate,
}, null, 2)}\n`);
