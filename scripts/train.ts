// 自己対戦で評価関数の重みを調整する（SPSA）。
//
// 学習の状態は training/ に置いてリポジトリで同期する。Mac でも GitHub Actions でも同じコマンドで動き、
//   1. 最新を pull
//   2. 制限時間まで SPSA で重みを更新
//   3. もう一度 pull して、自分が進めた分（重みの差分）を最新の状態に足し込む
//   4. 現チャンピオンと対戦させ、勝ち越したら新チャンピオンに昇格
//   5. commit して push（他所と競合したら 3 からやり直し）
// という流れなので、複数の場所で同時に回しても進捗が失われない。
//
// 例:
//   npm run train -- --minutes 30         # 30 分回して同期
//   npm run train -- --loop               # Ctrl-C まで 30 分ごとに同期しながら回し続ける
//   npm run train -- --minutes 5 --no-sync  # git 操作をせず手元だけで試す

import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { FEATURE_NAMES, type Feature, type Weights, withDefaults } from '../src/engine/ai';
import { PairPool } from './lib/pool';

const { values: args } = parseArgs({
  options: {
    minutes: { type: 'string', default: '30' },
    loop: { type: 'boolean', default: false },
    sync: { type: 'boolean', default: true },
    'no-sync': { type: 'boolean', default: false },
    source: { type: 'string' },
    threads: { type: 'string' },
  },
});

const SYNC = args.sync && !args['no-sync'];
// 公開リポジトリに残るので、マシン名などは入れず汎用的なラベルにする
const SOURCE = args.source ?? (process.env.GITHUB_ACTIONS ? 'actions' : 'mac');

// ---- ハイパーパラメータ ----
const TRAIN_DEPTH = 2;
// 最善から 0.1 歩以内の手をランダムに選ぶ。弱い手は選ばずに棋譜の多様性を確保する
const MARGIN = 1;
const OPENING_PLIES = 0;
const PAIRS_PER_ITER = 16;
const GATE_DEPTH = 3;
const GATE_PAIRS = 200;
const GATE_THRESHOLD = 0.55;
// pathDiff は「1 歩 = 10 点」という評価の単位として固定し、他の重みをそれに対して調整する
const FIXED: Feature[] = ['pathDiff'];
const TUNED = FEATURE_NAMES.filter((f) => !FIXED.includes(f));

// ---- 保存データ ----
const DIR = join(import.meta.dirname, '..', 'training');
const STATE = join(DIR, 'state.json');
const CHAMPION = join(DIR, 'champion.json');
const HISTORY = join(DIR, 'history.jsonl');

interface State {
  iteration: number;
  pairs: number;
  theta: Weights;
  updatedAt: string;
}

interface Champion {
  generation: number;
  weights: Weights;
  promotedAt: string;
  source: string;
  // 前のチャンピオンに対する勝率
  gate: { score: number; pairs: number; depth: number } | null;
  // 学習前の初期の重みに対する勝率（強くなっているかの物差し）
  vsInitial?: number;
}

const readJson = <T>(path: string): T => JSON.parse(readFileSync(path, 'utf8')) as T;
const writeJson = (path: string, v: unknown) => writeFileSync(path, `${JSON.stringify(v, null, 2)}\n`);
const round = (w: Weights): Weights =>
  Object.fromEntries(FEATURE_NAMES.map((k) => [k, Math.round(w[k] * 1000) / 1000])) as Weights;

function loadState(): State {
  if (existsSync(STATE)) {
    const s = readJson<State>(STATE);
    return { ...s, theta: withDefaults(s.theta) };
  }
  return { iteration: 0, pairs: 0, theta: withDefaults(), updatedAt: new Date().toISOString() };
}

function loadChampion(): Champion {
  if (existsSync(CHAMPION)) {
    const c = readJson<Champion>(CHAMPION);
    return { ...c, weights: withDefaults(c.weights) };
  }
  return { generation: 0, weights: withDefaults(), promotedAt: new Date().toISOString(), source: 'initial', gate: null };
}

// ---- git ----
const git = (...a: string[]) => execFileSync('git', a, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();

function pull() {
  if (!SYNC) return;
  git('pull', '--rebase', '--autostash', '--quiet', 'origin', 'main');
}

function ensureCleanTraining() {
  if (!SYNC) return;
  const branch = git('rev-parse', '--abbrev-ref', 'HEAD');
  if (branch !== 'main') throw new Error(`main ブランチで実行してください（現在: ${branch}）`);
  if (git('status', '--porcelain', '--', 'training')) {
    throw new Error('training/ に未コミットの変更があります。commit するか破棄してから実行してください');
  }
}

function commitAndPush(message: string): boolean {
  if (!SYNC) return true;
  git('add', '--', 'training');
  git('commit', '--quiet', '-m', message, '--', 'training');
  try {
    git('push', '--quiet', 'origin', 'HEAD:main');
    return true;
  } catch {
    // 他所が先に push した。自分のコミットを取り消して最新から足し直す
    git('reset', '--quiet', 'HEAD~1');
    git('checkout', '--', 'training');
    git('clean', '-fdq', '--', 'training');
    return false;
  }
}

// ---- SPSA ----
// 揺らす幅は特徴量ごとの固定値（重みに比例させると値が暴走する）。
// 目安は「その特徴量が典型的に 1 動いたとき、1 歩（10 点）の何割くらいか」。
const PERTURB: Record<Feature, number> = {
  pathDiff: 0,
  rowDiff: 1,
  wallDiff: 1,
  myWalls: 0.5,
  urgency: 4,
  wallsWhenClose: 2,
  center: 1,
  adjacent: 2,
};
const perturbScale = (_theta: Weights, k: Feature) => PERTURB[k];
const learningRate = (iteration: number) => 0.5 / (1 + iteration / 5000) ** 0.6;
const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

async function spsaStep(pool: PairPool, theta: Weights, iteration: number): Promise<Weights> {
  const delta = Object.fromEntries(TUNED.map((k) => [k, Math.random() < 0.5 ? -1 : 1])) as Record<Feature, number>;
  const plus = { ...theta };
  const minus = { ...theta };
  for (const k of TUNED) {
    plus[k] += perturbScale(theta, k) * delta[k];
    minus[k] -= perturbScale(theta, k) * delta[k];
  }
  const scores = await pool.runAll(
    Array.from({ length: PAIRS_PER_ITER }, () => ({
      a: plus,
      b: minus,
      opts: { depth: TRAIN_DEPTH, margin: MARGIN, openingPlies: OPENING_PLIES, seed: randomSeed() },
    })),
  );
  // plus 側の勝ち越し具合（-1〜1）
  const r = scores.reduce((x, y) => x + y, 0) / PAIRS_PER_ITER - 1;
  const next = { ...theta };
  const lr = learningRate(iteration);
  for (const k of TUNED) next[k] += lr * perturbScale(theta, k) * r * delta[k];
  return next;
}

// a の b に対する勝率（0〜1）
async function gate(pool: PairPool, a: Weights, b: Weights): Promise<number> {
  const scores = await pool.runAll(
    Array.from({ length: GATE_PAIRS }, (_, i) => ({
      a,
      b,
      // 判定用の序盤は固定シードにして、毎回同じ条件で比べる
      opts: { depth: GATE_DEPTH, margin: MARGIN, openingPlies: OPENING_PLIES, seed: 1_000_000 + i },
    })),
  );
  return scores.reduce((x, y) => x + y, 0) / (GATE_PAIRS * 2);
}

const fmt = (w: Weights) => TUNED.map((k) => `${k}=${w[k].toFixed(2)}`).join(' ');

async function runOnce(pool: PairPool, minutes: number) {
  ensureCleanTraining();
  pull();
  const start = loadState();
  let theta = start.theta;
  let iters = 0;
  const deadline = Date.now() + minutes * 60_000;
  const startedAt = new Date().toISOString();
  console.log(`[${SOURCE}] 開始 iteration=${start.iteration} ${fmt(theta)}`);

  while (Date.now() < deadline) {
    theta = await spsaStep(pool, theta, start.iteration + iters);
    iters++;
    if (iters % 100 === 0) console.log(`  +${iters} iters  ${fmt(theta)}`);
  }

  for (let attempt = 1; attempt <= 5; attempt++) {
    pull();
    // 自分が進めた分だけを最新の状態に足し込む（同時に回っていた他所の進捗も残る）
    const latest = loadState();
    const merged = { ...latest.theta };
    for (const k of TUNED) merged[k] += theta[k] - start.theta[k];
    const state: State = {
      iteration: latest.iteration + iters,
      pairs: latest.pairs + iters * PAIRS_PER_ITER,
      theta: round(merged),
      updatedAt: new Date().toISOString(),
    };

    const champion = loadChampion();
    const score = await gate(pool, state.theta, champion.weights);
    const promoted = score >= GATE_THRESHOLD;
    const vsInitial = await gate(pool, state.theta, withDefaults());
    const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
    console.log(`  チャンピオン（第 ${champion.generation} 世代）に対する勝率 ${pct(score)}${promoted ? ' → 昇格' : ''}、初期の重みに対する勝率 ${pct(vsInitial)}`);

    mkdirSync(join(DIR, 'champions'), { recursive: true });
    writeJson(STATE, state);
    let generation = champion.generation;
    if (promoted) {
      generation++;
      const next: Champion = {
        generation,
        weights: state.theta,
        promotedAt: state.updatedAt,
        source: SOURCE,
        gate: { score: Math.round(score * 1000) / 1000, pairs: GATE_PAIRS, depth: GATE_DEPTH },
        vsInitial: Math.round(vsInitial * 1000) / 1000,
      };
      writeJson(CHAMPION, next);
      writeJson(join(DIR, 'champions', `gen-${String(generation).padStart(4, '0')}.json`), next);
    }
    appendFileSync(
      HISTORY,
      `${JSON.stringify({
        startedAt,
        finishedAt: state.updatedAt,
        source: SOURCE,
        iterations: iters,
        totalIterations: state.iteration,
        theta: state.theta,
        gateScore: Math.round(score * 1000) / 1000,
        vsInitial: Math.round(vsInitial * 1000) / 1000,
        championGeneration: generation,
        promoted,
      })}\n`,
    );

    const msg = promoted
      ? `train: 第 ${generation} 世代に昇格（${SOURCE}, 勝率 ${(score * 100).toFixed(1)}%）`
      : `train: +${iters} iterations（${SOURCE}, 累計 ${state.iteration}）`;
    if (commitAndPush(msg)) {
      console.log(`[${SOURCE}] ${SYNC ? 'push 完了' : '保存完了（同期なし）'}: ${msg}`);
      return;
    }
    console.log('  push が競合したので最新を取り込んでやり直します');
  }
  throw new Error('push に 5 回失敗しました');
}

const pool = new PairPool(args.threads ? Number(args.threads) : undefined);
console.log(`threads=${pool.size} sync=${SYNC}`);
try {
  do {
    await runOnce(pool, Number(args.minutes));
  } while (args.loop);
} finally {
  await pool.close();
}
