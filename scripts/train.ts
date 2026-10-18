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
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { FEATURE_NAMES, type Feature, type Weights, withDefaults } from '../src/engine/ai';
import { OPENING_DEPTH, OPENING_PLIES, OPENING_TOP } from '../src/train/match';
import { fitCalibration } from '../src/train/calibrate';
import { PairPool, RecordPool } from './lib/pool';

const { values: args } = parseArgs({
  options: {
    minutes: { type: 'string', default: '30' },
    loop: { type: 'boolean', default: false },
    sync: { type: 'boolean', default: true },
    'no-sync': { type: 'boolean', default: false },
    // 判定（チャンピオン・初期の重みとの対戦）をせず、学習した差分を足し込むだけにする。
    // 並列で回すときは判定役を 1 台に絞り、残りは学習に専念させる
    'no-gate': { type: 'boolean', default: false },
    source: { type: 'string' },
    threads: { type: 'string' },
  },
});

const SYNC = args.sync && !args['no-sync'];
const GATE = !args['no-gate'];
// 公開リポジトリに残るので、マシン名などは入れず汎用的なラベルにする
const SOURCE = args.source ?? (process.env.GITHUB_ACTIONS ? 'actions' : 'mac');

// ---- ハイパーパラメータ ----
// 学習・判定・Web の「つよい」をすべて 3 手読みにそろえる。
// 重みの良し悪しは読みの深さで大きく変わるので、使う深さで鍛えて使う深さで判定する
const DEPTH = 3;
// 最善から 0.1 歩以内の手をランダムに選ぶ。弱い手は選ばずに棋譜の多様性を確保する
const MARGIN = 1;
const PAIRS_PER_ITER = 16;
// 判定に使う読みの深さと対局数（深さごとに測って平均する）
const GATE_SUITE: Record<number, number> = { [DEPTH]: 200 };
// 判定条件の識別子（深さ・対局数・序盤のばらけさせ方）
const GATE_VERSION = `${Object.entries(GATE_SUITE).map(([d, n]) => `d${d}x${n}`).join('-')}-o${OPENING_PLIES}t${OPENING_TOP}d${OPENING_DEPTH}`;
// 昇格条件: 平均勝率が GATE_THRESHOLD 以上、どの深さでも GATE_FLOOR 以上、初期の重みへの勝率が下がっていない
const GATE_FLOOR = 0.42;
const INITIAL_TOLERANCE = 0.03;
const GATE_THRESHOLD = 0.55;
// 学習中の重みがチャンピオンにこれ未満しか勝てなければ、チャンピオンの重みから学習し直す
const RESET_BELOW = 0.47;
// 対戦相手の選び方: この確率で現チャンピオン、残りは過去の世代（初期の重みを含む）から無作為
const CHAMPION_RATE = 0.5;
// pathDiff は「1 歩 = 10 点」という評価の単位として固定し、他の重みをそれに対して調整する
const FIXED: Feature[] = ['pathDiff'];
const TUNED = FEATURE_NAMES.filter((f) => !FIXED.includes(f));

// ---- 保存データ ----
const DIR = join(import.meta.dirname, '..', 'training');
const STATE = join(DIR, 'state.json');
const CHAMPION = join(DIR, 'champion.json');
const HISTORY = join(DIR, 'history.jsonl');
const CALIBRATION = join(DIR, 'calibration.json');
// 昇格したときに、評価値を勝率に換算する式を作り直すための自己対戦の局数
const CALIBRATION_GAMES = 200;

interface State {
  iteration: number;
  // 学習率の減衰に使う反復数（リセットで 0 に戻る）
  stepIteration?: number;
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
  gate: { score: number; byDepth?: Record<string, number> } | null;
  // 学習前の初期の重みに対する勝率（強くなっているかの物差し）
  vsInitial?: number;
  vsInitialByDepth?: Record<string, number>;
  // vsInitial を測ったときの判定条件（条件が変わったら測り直す）
  gateVersion?: string;
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

// 過去のチャンピオン全員（初期の重みを含む）。学習の対戦相手に使う
function loadLeague(): Weights[] {
  const dir = join(DIR, 'champions');
  const past = existsSync(dir)
    ? readdirSync(dir)
        .filter((f) => f.endsWith('.json'))
        .map((f) => withDefaults(readJson<Champion>(join(dir, f)).weights))
    : [];
  return [withDefaults(), ...past];
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
  lockedRace: 5,
  // 1 歩分の重みでチャンピオンに 68% 勝つことを確かめてから学習に入れた
  wallThreat: 2,
  // 重み 5 でチャンピオンに 58.5% 勝つことを確かめてから学習に入れた
  routeDiversity: 2,
};
const perturbScale = (_theta: Weights, k: Feature) => PERTURB[k];
// 4 台のランナーが同じ始点から進んだ差分を合算するため、1 台で学習していたときの大きさに抑える。
// 2 では高速化後の判定 9 回中 6 回がリセットになり、対チャンピオン勝率も 7.8〜54.5% と不安定だった。
const learningRate = (iteration: number) => 0.5 / (1 + iteration / 5000) ** 0.6;
const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

// ＋側と－側を「同じ相手・同じ乱数」で戦わせ、勝率の差で重みを動かす。
// ＋と－を直接戦わせると、隣の自分にだけ強い方向へ流れる（じゃんけん型の堂々巡り）ので避ける。
async function spsaStep(
  pool: PairPool,
  theta: Weights,
  stepIteration: number,
  champion: Weights,
  league: Weights[],
): Promise<Weights> {
  const delta = Object.fromEntries(TUNED.map((k) => [k, Math.random() < 0.5 ? -1 : 1])) as Record<Feature, number>;
  const plus = { ...theta };
  const minus = { ...theta };
  for (const k of TUNED) {
    plus[k] += perturbScale(theta, k) * delta[k];
    minus[k] -= perturbScale(theta, k) * delta[k];
  }
  const matches = Array.from({ length: PAIRS_PER_ITER }, () => ({
    opponent: Math.random() < CHAMPION_RATE ? champion : league[Math.floor(Math.random() * league.length)],
    opts: { depth: DEPTH, margin: MARGIN, openingPlies: OPENING_PLIES, seed: randomSeed() },
  }));
  const [plusScores, minusScores] = await Promise.all(
    [plus, minus].map((w) => pool.runAll(matches.map((m) => ({ a: w, b: m.opponent, opts: m.opts })))),
  );
  const sum = (xs: number[]) => xs.reduce((x, y) => x + y, 0);
  // ＋側が－側よりどれだけ多く勝ったか（-1〜1）
  const r = (sum(plusScores) - sum(minusScores)) / (PAIRS_PER_ITER * 2);
  const next = { ...theta };
  const lr = learningRate(stepIteration);
  for (const k of TUNED) next[k] += lr * perturbScale(theta, k) * r * delta[k];
  return next;
}

interface MatchResult {
  avg: number;
  byDepth: Record<string, number>;
}

// a の b に対する勝率（0〜1）を深さごとに測って平均する
async function gate(pool: PairPool, a: Weights, b: Weights): Promise<MatchResult> {
  const byDepth: Record<string, number> = {};
  for (const [depth, pairs] of Object.entries(GATE_SUITE)) {
    const scores = await pool.runAll(
      Array.from({ length: pairs }, (_, i) => ({
        a,
        b,
        // 判定用の序盤は固定シードにして、毎回同じ条件で比べる
        opts: { depth: Number(depth), margin: MARGIN, openingPlies: OPENING_PLIES, seed: 1_000_000 + i },
      })),
    );
    byDepth[depth] = round3(scores.reduce((x, y) => x + y, 0) / (pairs * 2));
  }
  const values = Object.values(byDepth);
  return { avg: round3(values.reduce((x, y) => x + y, 0) / values.length), byDepth };
}

const round3 = (x: number) => Math.round(x * 1000) / 1000;
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const fmtMatch = (m: MatchResult) =>
  `${pct(m.avg)}（${Object.entries(m.byDepth).map(([d, v]) => `${d}手読み ${pct(v)}`).join(' / ')}）`;

async function refreshCalibration() {
  const champ = loadChampion();
  const current = () => (existsSync(CALIBRATION) ? readJson<{ generation: number }>(CALIBRATION) : null);
  if (champ.generation === 0 || current()?.generation === champ.generation) return;
  const cal = await calibrate(champ.weights, champ.generation);
  for (let attempt = 1; attempt <= 5; attempt++) {
    pull();
    // 待っている間に昇格していたら、その式は使えないので次の回に任せる
    if (loadChampion().generation !== champ.generation || current()?.generation === champ.generation) return;
    writeJson(CALIBRATION, cal);
    if (commitAndPush(`train: 第 ${champ.generation} 世代の勝率換算式を作成（${SOURCE}）`)) return;
  }
}

// 新しいチャンピオンの評価値を勝率に換算する式（Web の形勢表示で使う）
async function calibrate(weights: Weights, generation: number) {
  const records = new RecordPool(pool.size);
  try {
    const games = await records.runAll(
      Array.from({ length: CALIBRATION_GAMES }, (_, i) => ({ weights, depth: DEPTH, seed: 31_000_000 + i })),
    );
    const cal = fitCalibration(games, generation);
    console.log(`  勝率への換算式: a=${cal.a.toPrecision(4)} b=${cal.b.toFixed(3)}`);
    return cal;
  } finally {
    await records.close();
  }
}

const fmt = (w: Weights) => TUNED.map((k) => `${k}=${w[k].toFixed(2)}`).join(' ');

async function runOnce(pool: PairPool, minutes: number) {
  ensureCleanTraining();
  pull();
  // 換算式がチャンピオンの世代と合っていなければ（昇格が換算式を作る前のコードだった場合など）作り直す。
  // 判定役だけが行い、push が競合したら最新を取り込んで保存し直す（作った式を捨てない）
  if (GATE) await refreshCalibration();
  const start = loadState();
  const startStep = start.stepIteration ?? start.iteration;
  const champion0 = loadChampion().weights;
  const league = loadLeague();
  let theta = start.theta;
  let iters = 0;
  const deadline = Date.now() + minutes * 60_000;
  const startedAt = new Date().toISOString();
  console.log(`[${SOURCE}] 開始 iteration=${start.iteration} ${fmt(theta)}`);

  while (Date.now() < deadline) {
    theta = await spsaStep(pool, theta, startStep + iters, champion0, league);
    iters++;
    if (iters % 100 === 0) console.log(`  +${iters} iters  ${fmt(theta)}`);
  }

  for (let attempt = 1; attempt <= 10; attempt++) {
    pull();
    // 自分が進めた分だけを最新の状態に足し込む（同時に回っていた他所の進捗も残る）
    const latest = loadState();
    const merged = { ...latest.theta };
    for (const k of TUNED) merged[k] += theta[k] - start.theta[k];
    const state: State = {
      iteration: latest.iteration + iters,
      stepIteration: (latest.stepIteration ?? latest.iteration) + iters,
      pairs: latest.pairs + iters * PAIRS_PER_ITER * 2,
      theta: round(merged),
      updatedAt: new Date().toISOString(),
    };

    const champion = loadChampion();
    let vsChampion: MatchResult | null = null;
    let vsInitial: MatchResult | null = null;
    let promoted = false;
    let reset = false;
    if (GATE) {
      // 判定の条件が変わっていたら、チャンピオンの対初期勝率を今の条件で測り直す
      if (champion.gateVersion !== GATE_VERSION) {
        champion.gateVersion = GATE_VERSION;
        const m = await gate(pool, champion.weights, withDefaults());
        champion.vsInitial = m.avg;
        champion.vsInitialByDepth = m.byDepth;
        if (champion.generation > 0) writeJson(CHAMPION, champion);
      }
      vsChampion = await gate(pool, state.theta, champion.weights);
      vsInitial = await gate(pool, state.theta, withDefaults());
      const score = vsChampion.avg;
      promoted =
        score >= GATE_THRESHOLD &&
        Math.min(...Object.values(vsChampion.byDepth)) >= GATE_FLOOR &&
        vsInitial.avg >= (champion.vsInitial ?? 0.5) - INITIAL_TOLERANCE;
      console.log(`  対チャンピオン（第 ${champion.generation} 世代）${fmtMatch(vsChampion)}`);
      console.log(`  対初期の重み ${fmtMatch(vsInitial)}（チャンピオンは ${pct(champion.vsInitial ?? 0.5)}）${promoted ? ' → 昇格' : ''}`);

      reset = !promoted && score < RESET_BELOW;
      if (reset) {
        console.log('  チャンピオンに大きく負け越したので、チャンピオンの重みから学習し直します');
        state.theta = champion.weights;
        state.stepIteration = 0;
      }
    }

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
        gate: { score: vsChampion!.avg, byDepth: vsChampion!.byDepth },
        vsInitial: vsInitial!.avg,
        vsInitialByDepth: vsInitial!.byDepth,
        gateVersion: GATE_VERSION,
      };
      writeJson(CHAMPION, next);
      writeJson(join(DIR, 'champions', `gen-${String(generation).padStart(4, '0')}.json`), next);
      writeJson(CALIBRATION, await calibrate(next.weights, generation));
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
        gateScore: vsChampion?.avg,
        gateByDepth: vsChampion?.byDepth,
        vsInitial: vsInitial?.avg,
        vsInitialByDepth: vsInitial?.byDepth,
        championGeneration: generation,
        promoted,
        reset,
      })}\n`,
    );

    const msg = promoted
      ? `train: 第 ${generation} 世代に昇格（${SOURCE}, 勝率 ${(vsChampion!.avg * 100).toFixed(1)}%）`
      : `train: +${iters} iterations（${SOURCE}, 累計 ${state.iteration}）`;
    if (commitAndPush(msg)) {
      console.log(`[${SOURCE}] ${SYNC ? 'push 完了' : '保存完了（同期なし）'}: ${msg}`);
      return;
    }
    console.log('  push が競合したので最新を取り込んでやり直します');
  }
  throw new Error('push に 10 回失敗しました');
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
