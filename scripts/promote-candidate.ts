// Experimentで見つけた候補をTrainと同じ基準で再検証し、通過時だけ新世代へ昇格する。
// training/を書き換えるため、Promote Candidateワークフロー以外では実行しない。

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { FEATURE_NAMES, WIDE_ROOT_DEFAULT, type Weights, withDefaults } from '../src/engine/ai';
import { fitCalibration } from '../src/train/calibrate';
import { OPENING_DEPTH, OPENING_PLIES, OPENING_TOP } from '../src/train/match';
import { PairPool, RecordPool } from './lib/pool';

const { values: args } = parseArgs({
  options: {
    candidate: { type: 'string' },
    source: { type: 'string', default: 'actions-promote' },
  },
});
if (!args.candidate) throw new Error('--candidate を指定してください');
if (!/^experiments\/candidates\/[a-z0-9-]+\.json$/.test(args.candidate)) {
  throw new Error('candidate は experiments/candidates/ 内のJSONを指定してください');
}

const root = join(import.meta.dirname, '..');
const trainingDir = join(root, 'training');
const candidatePath = join(root, args.candidate);
const candidateFile = JSON.parse(readFileSync(candidatePath, 'utf8')) as {
  expectedChampionGeneration: number;
  source: string;
  evidence?: unknown;
  weights: Partial<Weights>;
};
interface Champion {
  generation: number;
  weights: Weights;
  promotedAt: string;
  source: string;
  gate: { score: number; byDepth?: Record<string, number> } | null;
  vsInitial?: number;
  vsInitialByDepth?: Record<string, number>;
  gateVersion?: string;
}
interface State {
  iteration: number;
  stepIteration?: number;
  featureSeedVersion?: number;
  pairs: number;
  theta: Weights;
  updatedAt: string;
}

const readJson = <T>(path: string): T => JSON.parse(readFileSync(path, 'utf8')) as T;
const writeJson = (path: string, value: unknown) => writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
const round3 = (value: number) => Math.round(value * 1000) / 1000;
const roundWeights = (weights: Weights): Weights => Object.fromEntries(
  FEATURE_NAMES.map((feature) => [feature, round3(weights[feature])]),
) as Weights;
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

const championPath = join(trainingDir, 'champion.json');
const statePath = join(trainingDir, 'state.json');
const historyPath = join(trainingDir, 'history.jsonl');
const calibrationPath = join(trainingDir, 'calibration.json');
const champion = readJson<Champion>(championPath);
const state = readJson<State>(statePath);
if (champion.generation !== candidateFile.expectedChampionGeneration) {
  throw new Error(`チャンピオン世代が変わっています: expected=${candidateFile.expectedChampionGeneration} actual=${champion.generation}`);
}
const candidate = roundWeights(withDefaults(candidateFile.weights));
const initial = withDefaults();
const depth = 3;
const margin = 1;
const gatePairs = 800;
const initialPairs = 200;
const gateThreshold = 0.525;
const initialTolerance = 0.03;
const gateVersion = `d${depth}x200-confirm800t525-o${OPENING_PLIES}t${OPENING_TOP}d${OPENING_DEPTH}-r${WIDE_ROOT_DEFAULT ? 'wide' : 'narrow'}`;

async function compare(pool: PairPool, a: Weights, b: Weights, pairs: number, seed: number) {
  const scores = await pool.runAll(Array.from({ length: pairs }, (_, index) => ({
    a,
    b,
    opts: { depth, margin, openingPlies: OPENING_PLIES, seed: seed + index },
  })));
  return round3(scores.reduce((sum, score) => sum + score, 0) / (pairs * 2));
}

const pool = new PairPool();
let vsChampion: number;
let vsInitial: number;
try {
  vsChampion = await compare(pool, candidate, withDefaults(champion.weights), gatePairs, 220_000_000);
  vsInitial = await compare(pool, candidate, initial, initialPairs, 221_000_000);
} finally {
  await pool.close();
}
console.log(`対第${champion.generation}世代: ${pct(vsChampion)}（${gatePairs * 2}局）`);
console.log(`対初期: ${pct(vsInitial)}（${initialPairs * 2}局、現チャンピオン ${pct(champion.vsInitial ?? 0.5)}）`);
if (vsChampion < gateThreshold || vsInitial < (champion.vsInitial ?? 0.5) - initialTolerance) {
  throw new Error('昇格基準を満たしませんでした。training/は変更していません');
}

const generation = champion.generation + 1;
const records = new RecordPool();
let calibration;
try {
  const games = await records.runAll(Array.from({ length: 200 }, (_, index) => ({
    weights: candidate,
    depth,
    seed: 31_000_000 + index,
  })));
  calibration = fitCalibration(games, generation);
} finally {
  await records.close();
}

const promotedAt = new Date().toISOString();
const nextChampion: Champion = {
  generation,
  weights: candidate,
  promotedAt,
  source: args.source!,
  gate: { score: vsChampion, byDepth: { [depth]: vsChampion } },
  vsInitial,
  vsInitialByDepth: { [depth]: vsInitial },
  gateVersion,
};
const nextState: State = { ...state, theta: candidate, updatedAt: promotedAt };
mkdirSync(join(trainingDir, 'champions'), { recursive: true });
writeJson(statePath, nextState);
writeJson(championPath, nextChampion);
writeJson(join(trainingDir, 'champions', `gen-${String(generation).padStart(4, '0')}.json`), nextChampion);
writeJson(calibrationPath, calibration);
appendFileSync(historyPath, `${JSON.stringify({
  startedAt: promotedAt,
  finishedAt: promotedAt,
  source: args.source,
  iterations: 0,
  totalIterations: state.iteration,
  theta: candidate,
  gateScore: vsChampion,
  gateByDepth: { [depth]: vsChampion },
  vsInitial,
  vsInitialByDepth: { [depth]: vsInitial },
  championGeneration: generation,
  promoted: true,
  reset: false,
  pairsPerIteration: 64,
  perturbMultiplier: 0.5,
  learningRateBase: 0.25,
  candidateSource: candidateFile.source,
  candidateEvidence: candidateFile.evidence,
})}\n`);
console.log(`第${generation}世代へ昇格。勝率換算式 a=${calibration.a.toPrecision(4)} b=${calibration.b.toFixed(3)}`);
