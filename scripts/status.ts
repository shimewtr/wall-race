// 学習の進み具合を表示する。重みは「1 歩 = 10 点」で割って「何歩分の価値か」で見せる。

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DEFAULT_WEIGHTS, FEATURES, FEATURE_NAMES, type Weights, withDefaults } from '../src/engine/ai';

const DIR = join(import.meta.dirname, '..', 'training');

interface Run {
  finishedAt: string;
  source: string;
  iterations: number;
  totalIterations: number;
  theta: Weights;
  gateScore: number;
  vsInitial?: number;
  championGeneration: number;
  promoted: boolean;
}

if (!existsSync(join(DIR, 'state.json'))) {
  console.log('まだ学習していません（npm run train で開始）');
  process.exit(0);
}

const state = JSON.parse(readFileSync(join(DIR, 'state.json'), 'utf8'));
const champion = existsSync(join(DIR, 'champion.json'))
  ? JSON.parse(readFileSync(join(DIR, 'champion.json'), 'utf8'))
  : { generation: 0, weights: DEFAULT_WEIGHTS };
const runs: Run[] = readFileSync(join(DIR, 'history.jsonl'), 'utf8')
  .split('\n')
  .filter(Boolean)
  .map((l) => JSON.parse(l));

const pct = (x: number | undefined) => (x === undefined ? '   -  ' : `${(x * 100).toFixed(1).padStart(5)}%`);
const time = (iso: string) =>
  new Date(iso).toLocaleString('ja-JP', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });

console.log(`チャンピオン: 第 ${champion.generation} 世代${champion.vsInitial !== undefined ? `（初期の重みに対する勝率 ${pct(champion.vsInitial).trim()}）` : ''}`);
console.log(`累計: ${state.iteration} 反復 / ${state.pairs * 2} 局\n`);

const cw = withDefaults(champion.weights);
const tw = withDefaults(state.theta);
console.log('特徴量            初期     チャンピオン  学習中   （単位: 歩）');
for (const k of FEATURE_NAMES) {
  const f = (w: Weights) => (w[k] / 10).toFixed(2).padStart(8);
  console.log(`${k.padEnd(16)}${f(DEFAULT_WEIGHTS)}${f(cw)}    ${f(tw)}   ${FEATURES[k]}`);
}

const bySource = new Map<string, { runs: number; iterations: number }>();
for (const r of runs) {
  const s = bySource.get(r.source) ?? { runs: 0, iterations: 0 };
  s.runs++;
  s.iterations += r.iterations;
  bySource.set(r.source, s);
}
console.log(`\n実行場所: ${[...bySource].map(([k, v]) => `${k} ${v.runs} 回 / ${v.iterations} 反復`).join('、')}`);

console.log('\n直近の実行:');
console.log('  終了          場所      反復   対チャンピオン  対初期   世代');
for (const r of runs.slice(-10)) {
  console.log(
    `  ${time(r.finishedAt).padEnd(12)}  ${r.source.padEnd(8)}${String(r.iterations).padStart(6)}   ${pct(r.gateScore)}        ${pct(r.vsInitial)}  ${r.championGeneration}${r.promoted ? ' ↑昇格' : ''}`,
  );
}
