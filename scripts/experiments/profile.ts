// 探索のどの関数に時間がかかっているかを測る（CPU プロファイル）。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/profile.ts --depth 3

import { existsSync, readFileSync } from 'node:fs';
import { Session } from 'node:inspector/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { detourStats, search, withDefaults } from '../../src/engine/ai';
import { type GameState, applyMove, distanceStats, initialState, isOver } from '../../src/engine/game';
import { rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({ options: { depth: { type: 'string', default: '3' } } });

const CHAMPION = join(import.meta.dirname, '..', '..', 'training', 'champion.json');
const weights = withDefaults(existsSync(CHAMPION) ? JSON.parse(readFileSync(CHAMPION, 'utf8')).weights : {});

const positions: GameState[] = [];
for (let g = 0; g < 6; g++) {
  const rng = rngFrom(21_000 + g);
  let s = initialState();
  while (!isOver(s)) {
    positions.push(s);
    s = applyMove(s, search(s, { maxDepth: 2, weights, margin: 10, rng }).move);
  }
}
const sample = positions.filter((_, i) => i % 4 === 0).slice(0, 80);

const session = new Session();
session.connect();
await session.post('Profiler.enable');
await session.post('Profiler.setSamplingInterval', { interval: 100 });
await session.post('Profiler.start');
const t = performance.now();
for (const s of sample) search(s, { maxDepth: Number(args.depth), weights });
const total = performance.now() - t;
const { profile } = await session.post('Profiler.stop');

// 関数ごとの自己時間（その関数自身で使った時間）
const self = new Map<string, number>();
const dt = profile.timeDeltas ?? [];
const hits = new Map<number, number>();
(profile.samples ?? []).forEach((id, i) => hits.set(id, (hits.get(id) ?? 0) + (dt[i] ?? 0)));
for (const n of profile.nodes) {
  const f = n.callFrame;
  const file = f.url.split('/').slice(-2).join('/');
  const key = `${f.functionName || '(anonymous)'} ${file}:${f.lineNumber + 1}`;
  self.set(key, (self.get(key) ?? 0) + (hits.get(n.id) ?? 0));
}
const sum = [...self.values()].reduce((a, b) => a + b, 0);
console.log(`maxDetour: ${detourStats.calls} 回呼ばれ、キャッシュに無く計算したのは ${detourStats.misses} 回（当たり率 ${((1 - detourStats.misses / detourStats.calls) * 100).toFixed(1)}%）`);
console.log(`距離表: ${distanceStats.calls} 回参照され、キャッシュに無く計算したのは ${distanceStats.misses} 回（当たり率 ${((1 - distanceStats.misses / distanceStats.calls) * 100).toFixed(1)}%）`);
console.log(`${sample.length} 局面を ${args.depth} 手読み: 合計 ${total.toFixed(0)}ms\n`);
console.log('自己時間の多い関数:');
for (const [k, v] of [...self].sort((a, b) => b[1] - a[1]).slice(0, 20)) {
  console.log(`  ${((v / sum) * 100).toFixed(1).padStart(5)}%  ${k}`);
}

// 重い関数がどこから呼ばれているか（呼び出し元ごとの、その関数の自己時間）
const byId = new Map(profile.nodes.map((n) => [n.id, n]));
const parent = new Map<number, number>();
for (const n of profile.nodes) for (const c of n.children ?? []) parent.set(c, n.id);
const name = (id: number | undefined) => (id === undefined ? '(root)' : byId.get(id)!.callFrame.functionName || '(anonymous)');
for (const target of ['shortestDistance', 'canStep', 'shortestPath', 'goalDistances']) {
  const callers = new Map<string, number>();
  for (const n of profile.nodes) {
    if (n.callFrame.functionName !== target) continue;
    // canStep は呼び出し元の呼び出し元まで見る（canStep 自体はほぼ BFS から呼ばれる）
    let p = parent.get(n.id);
    if (target === 'canStep') p = parent.get(p!);
    const key = target === 'canStep' ? `${name(parent.get(n.id))} ← ${name(p)}` : name(p);
    callers.set(key, (callers.get(key) ?? 0) + (hits.get(n.id) ?? 0));
  }
  const t = [...callers.values()].reduce((a, b) => a + b, 0);
  console.log(`\n${target} の呼び出し元:`);
  for (const [k, v] of [...callers].sort((a, b) => b[1] - a[1]).slice(0, 6)) {
    console.log(`  ${((v / t) * 100).toFixed(1).padStart(5)}%  ${k}`);
  }
}
