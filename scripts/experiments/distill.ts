// 3手読みと4手読みで最善手が変わる局面を、小さな決定木へ蒸留する。
// 重い局面収集は GitHub Actions の Experiment ワークフローで行う。
//
// 例: npx tsx scripts/experiments/distill.ts --train 800 --test 800 --games 70 --max-depth 3

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { recentCells, search } from '../../src/engine/ai';
import {
  WALL_SLOTS,
  type GameState,
  type Player,
  applyMove,
  cloneState,
  initialState,
  isOver,
  moveToString,
  shortestDistance,
} from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({
  options: {
    train: { type: 'string', default: '800' },
    test: { type: 'string', default: '800' },
    games: { type: 'string', default: '70' },
    'sample-every': { type: 'string', default: '4' },
    'max-depth': { type: 'string', default: '3' },
    'min-leaf': { type: 'string', default: '30' },
    'train-seed': { type: 'string', default: '95000000' },
    'test-seed': { type: 'string', default: '96000000' },
  },
});

const champion = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'));
const trainLimit = Number(args.train);
const testLimit = Number(args.test);
const games = Number(args.games);
const sampleEvery = Number(args['sample-every']);
const maxDepth = Number(args['max-depth']);
const minLeaf = Number(args['min-leaf']);
const trainSeed = Number(args['train-seed']);
const testSeed = Number(args['test-seed']);
const other = (p: Player): Player => (1 - p) as Player;

const FEATURE_LABELS = {
  ply: '手数',
  myDist: '自分の残り歩数',
  opDist: '相手の残り歩数',
  distGap: '歩数差（相手−自分）',
  myWalls: '自分の残り壁',
  opWalls: '相手の残り壁',
  wallGap: '壁枚数差（自分−相手）',
  totalWalls: '盤上の壁数',
  connectedWalls: '連結している壁数',
  maxMyDepend: '自分経路の最大壁依存',
  maxOpDepend: '相手経路の最大壁依存',
  dependGap: '最大壁依存差（相手−自分）',
  myCriticalWalls: '自分経路に効く既存壁数',
  opCriticalWalls: '相手経路に効く既存壁数',
  sumMyDepend: '自分経路の壁依存合計',
  sumOpDepend: '相手経路の壁依存合計',
} as const;
type FeatureName = keyof typeof FEATURE_LABELS;
type Values = Record<FeatureName, number>;
interface Row { values: Values; label: boolean }

const BASIC_FEATURES: FeatureName[] = ['ply', 'myDist', 'opDist', 'distGap', 'myWalls', 'opWalls', 'wallGap', 'totalWalls'];
const ALL_FEATURES = Object.keys(FEATURE_LABELS) as FeatureName[];

function wallAt(walls: Uint8Array, x: number, y: number): boolean {
  return x >= 0 && x < WALL_SLOTS && y >= 0 && y < WALL_SLOTS && walls[y * WALL_SLOTS + x] === 1;
}

function values(state: GameState): Values {
  const me = state.turn;
  const op = other(me);
  const myDist = shortestDistance(state, me);
  const opDist = shortestDistance(state, op);
  let totalWalls = 0;
  let connectedWalls = 0;
  let maxMyDepend = 0;
  let maxOpDepend = 0;
  let myCriticalWalls = 0;
  let opCriticalWalls = 0;
  let sumMyDepend = 0;
  let sumOpDepend = 0;

  for (const o of ['h', 'v'] as const) {
    const walls = o === 'h' ? state.hWalls : state.vWalls;
    for (let y = 0; y < WALL_SLOTS; y++) {
      for (let x = 0; x < WALL_SLOTS; x++) {
        if (!wallAt(walls, x, y)) continue;
        totalWalls++;
        const connected = o === 'h'
          ? wallAt(walls, x - 2, y) || wallAt(walls, x + 2, y)
          : wallAt(walls, x, y - 2) || wallAt(walls, x, y + 2);
        if (connected) connectedWalls++;
        const edited = cloneState(state);
        (o === 'h' ? edited.hWalls : edited.vWalls)[y * WALL_SLOTS + x] = 0;
        const myDepend = myDist - shortestDistance(edited, me);
        const opDepend = opDist - shortestDistance(edited, op);
        maxMyDepend = Math.max(maxMyDepend, myDepend);
        maxOpDepend = Math.max(maxOpDepend, opDepend);
        if (myDepend > 0) myCriticalWalls++;
        if (opDepend > 0) opCriticalWalls++;
        sumMyDepend += myDepend;
        sumOpDepend += opDepend;
      }
    }
  }
  return {
    ply: state.ply,
    myDist,
    opDist,
    distGap: opDist - myDist,
    myWalls: state.wallsLeft[me],
    opWalls: state.wallsLeft[op],
    wallGap: state.wallsLeft[me] - state.wallsLeft[op],
    totalWalls,
    connectedWalls,
    maxMyDepend,
    maxOpDepend,
    dependGap: maxOpDepend - maxMyDepend,
    myCriticalWalls,
    opCriticalWalls,
    sumMyDepend,
    sumOpDepend,
  };
}

function collect(seed: number, limit: number): Row[] {
  const rows: Row[] = [];
  for (let game = 0; game < games && rows.length < limit; game++) {
    const rng = rngFrom(seed + game);
    const history: GameState[] = [initialState()];
    let state = history[0];
    while (!isOver(state) && rows.length < limit) {
      if (state.ply % sampleEvery === game % sampleEvery) {
        const move3 = search(state, { maxDepth: 3, weights: champion.weights, repeatPenalty: 0 }).move;
        const move4 = search(state, { maxDepth: 4, weights: champion.weights, repeatPenalty: 0 }).move;
        rows.push({ values: values(state), label: moveToString(move3) !== moveToString(move4) });
      }
      const move = search(state, {
        maxDepth: depthAt(state.ply, 3, OPENING_PLIES),
        weights: champion.weights,
        margin: 1,
        pickFromTop: pickAt(state.ply, OPENING_PLIES),
        rng,
        recent: recentCells(history, state.turn),
      }).move;
      state = applyMove(state, move);
      history.push(state);
    }
  }
  return rows;
}

interface Leaf { kind: 'leaf'; count: number; positives: number; probability: number }
interface Split { kind: 'split'; feature: FeatureName; threshold: number; count: number; positives: number; left: Tree; right: Tree }
type Tree = Leaf | Split;

const positiveRate = (rows: Row[]) => rows.filter((row) => row.label).length / Math.max(1, rows.length);
function weightedGini(rows: Row[], positiveWeight: number): number {
  const positive = rows.filter((row) => row.label).length * positiveWeight;
  const negative = rows.length - rows.filter((row) => row.label).length;
  const total = positive + negative;
  if (total === 0) return 0;
  const p = positive / total;
  return 2 * p * (1 - p);
}
const weightedMass = (rows: Row[], positiveWeight: number) =>
  rows.reduce((sum, row) => sum + (row.label ? positiveWeight : 1), 0);

function buildTree(rows: Row[], featureNames: FeatureName[], depth: number, positiveWeight: number): Tree {
  const positives = rows.filter((row) => row.label).length;
  const leaf = (): Leaf => ({ kind: 'leaf', count: rows.length, positives, probability: positives / Math.max(1, rows.length) });
  if (depth >= maxDepth || rows.length < minLeaf * 2 || positives === 0 || positives === rows.length) return leaf();
  const parentImpurity = weightedGini(rows, positiveWeight);
  let best: { feature: FeatureName; threshold: number; gain: number; left: Row[]; right: Row[] } | null = null;
  for (const feature of featureNames) {
    const unique = [...new Set(rows.map((row) => row.values[feature]))].sort((a, b) => a - b);
    for (let i = 0; i + 1 < unique.length; i++) {
      const threshold = (unique[i] + unique[i + 1]) / 2;
      const left = rows.filter((row) => row.values[feature] <= threshold);
      const right = rows.filter((row) => row.values[feature] > threshold);
      if (left.length < minLeaf || right.length < minLeaf) continue;
      const parentMass = weightedMass(rows, positiveWeight);
      const weighted = (
        weightedMass(left, positiveWeight) * weightedGini(left, positiveWeight) +
        weightedMass(right, positiveWeight) * weightedGini(right, positiveWeight)
      ) / parentMass;
      const gain = parentImpurity - weighted;
      if (!best || gain > best.gain) best = { feature, threshold, gain, left, right };
    }
  }
  if (!best || best.gain <= 1e-9) return leaf();
  return {
    kind: 'split', feature: best.feature, threshold: best.threshold, count: rows.length, positives,
    left: buildTree(best.left, featureNames, depth + 1, positiveWeight),
    right: buildTree(best.right, featureNames, depth + 1, positiveWeight),
  };
}

function score(tree: Tree, row: Row): number {
  if (tree.kind === 'leaf') return tree.probability;
  return score(row.values[tree.feature] <= tree.threshold ? tree.left : tree.right, row);
}

function auc(rows: Row[], scores: number[]): number {
  let wins = 0;
  let pairs = 0;
  for (let i = 0; i < rows.length; i++) {
    if (!rows[i].label) continue;
    for (let j = 0; j < rows.length; j++) {
      if (rows[j].label) continue;
      pairs++;
      wins += scores[i] > scores[j] ? 1 : scores[i] === scores[j] ? 0.5 : 0;
    }
  }
  return wins / Math.max(1, pairs);
}

function printTree(tree: Tree, prefix = '') {
  if (tree.kind === 'leaf') {
    console.log(`${prefix}→ 変更 ${tree.positives}/${tree.count}（${(100 * tree.probability).toFixed(1)}%）`);
    return;
  }
  console.log(`${prefix}${FEATURE_LABELS[tree.feature]} ≤ ${tree.threshold.toFixed(2)}`);
  printTree(tree.left, `${prefix}  はい: `);
  printTree(tree.right, `${prefix}  いいえ: `);
}

function evaluateTree(name: string, tree: Tree, train: Row[], test: Row[]) {
  const threshold = positiveRate(train);
  const scores = test.map((row) => score(tree, row));
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < test.length; i++) {
    const predicted = scores[i] >= threshold;
    if (predicted && test[i].label) tp++;
    else if (predicted) fp++;
    else if (test[i].label) fn++;
    else tn++;
  }
  const precision = tp / Math.max(1, tp + fp);
  const recall = tp / Math.max(1, tp + fn);
  const specificity = tn / Math.max(1, tn + fp);
  console.log(`\n## ${name}`);
  printTree(tree);
  console.log(`- 未知シード: 適合率 ${(100 * precision).toFixed(1)}%、再現率 ${(100 * recall).toFixed(1)}%、特異度 ${(100 * specificity).toFixed(1)}%、balanced accuracy ${(50 * (recall + specificity)).toFixed(1)}%、AUC ${auc(test, scores).toFixed(3)}`);
  console.log(`- 適合率の基準比: ${(precision / positiveRate(test)).toFixed(2)}倍（テスト変更率 ${(100 * positiveRate(test)).toFixed(1)}%）`);
  const ranked = test.map((row, index) => ({ row, score: scores[index] })).sort((a, b) => b.score - a.score);
  for (const budget of [0.2, 0.3]) {
    const picked = ranked.slice(0, Math.ceil(test.length * budget));
    const caught = picked.filter((item) => item.row.label).length;
    const total = test.filter((row) => row.label).length;
    console.log(`- 危険度上位${budget * 100}%だけ4手読み: 判断差を ${caught}/${total}（${(100 * caught / Math.max(1, total)).toFixed(1)}%）捕捉`);
  }
}

const train = collect(trainSeed, trainLimit);
const test = collect(testSeed, testLimit);
const trainRate = positiveRate(train);
const positiveWeight = (1 - trainRate) / Math.max(1e-9, trainRate);
const basicTree = buildTree(train, BASIC_FEATURES, 0, positiveWeight);
const fullTree = buildTree(train, ALL_FEATURES, 0, positiveWeight);

console.log(`# 第 ${champion.generation} 世代の深読み条件蒸留`);
console.log();
console.log(`学習: シード${trainSeed}・${train.length}局面・変更${train.filter((row) => row.label).length}（${(100 * trainRate).toFixed(1)}%）`);
console.log(`評価: シード${testSeed}・${test.length}局面・変更${test.filter((row) => row.label).length}（${(100 * positiveRate(test)).toFixed(1)}%）`);
console.log(`木: 最大深さ${maxDepth}、最小葉${minLeaf}、変更群と安定群を均衡重み付け`);
evaluateTree('基本量だけ', basicTree, train, test);
evaluateTree('経路の壁依存を含む', fullTree, train, test);

console.log('\n## 読み方');
console.log('- 目的は4手読みを使う局面の絞り込みであり、この木を評価値へ直接加えるものではない。');
console.log('- 上位20〜30%で十分な判断差を拾え、基本量だけの木を未知シードで上回る場合に限り、速度と対戦勝率の検証へ進む。');
