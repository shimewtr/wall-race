// 3手読みと4手読みで判断が割れる局面の前後を記号化し、安定局面より多い手順を探す。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/motifs.ts --games 80 --positions 600 --sample-every 4

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { recentCells, search } from '../../src/engine/ai';
import {
  WALL_SLOTS,
  type GameState,
  type Move,
  type Player,
  applyMove,
  initialState,
  isOver,
  moveToString,
  shortestDistance,
} from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '80' },
    positions: { type: 'string', default: '600' },
    'sample-every': { type: 'string', default: '4' },
    context: { type: 'string', default: '4' },
    'min-count': { type: 'string', default: '5' },
    examples: { type: 'string', default: '10' },
    seed: { type: 'string', default: '91000000' },
  },
});

const champion = JSON.parse(
  readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'),
);
const games = Number(args.games);
const maxPositions = Number(args.positions);
const sampleEvery = Number(args['sample-every']);
const context = Number(args.context);
const minCount = Number(args['min-count']);
const maxExamples = Number(args.examples);
const seed = Number(args.seed);

interface RecordedGame {
  states: GameState[];
  moves: Move[];
}

interface Observation {
  game: number;
  ply: number;
  phase: string;
  changed: boolean;
  move3: string;
  move4: string;
  token3: string;
  token4: string;
  position: string;
  coarse: string[];
  tactical: string[];
}

const other = (player: Player): Player => (1 - player) as Player;
const phaseAt = (ply: number) => (ply < 20 ? '序盤' : ply < 40 ? '中盤' : '終盤');
const sign = (value: number) => (value > 0 ? '+' : value < 0 ? '-' : '0');
const hasWall = (walls: Uint8Array, x: number, y: number) =>
  x >= 0 && x < WALL_SLOTS && y >= 0 && y < WALL_SLOTS && walls[y * WALL_SLOTS + x] === 1;

// C は同じ向きの壁へ端をそろえて延長する手。+/- は相手の遠回り - 自分の遠回りの符号。
function moveToken(state: GameState, move: Move, detailed: boolean): string {
  if (!detailed) return move.type === 'pawn' ? 'P' : move.o.toUpperCase();
  const me = state.turn;
  const beforeMe = shortestDistance(state, me);
  const beforeOp = shortestDistance(state, other(me));
  const after = applyMove(state, move);
  if (move.type === 'pawn') return `P${sign(beforeMe - shortestDistance(after, me))}`;

  const afterMe = shortestDistance(after, me);
  const afterOp = shortestDistance(after, other(me));
  const net = (afterOp - beforeOp) - (afterMe - beforeMe);
  const connected = move.o === 'h'
    ? hasWall(state.hWalls, move.x - 2, move.y) || hasWall(state.hWalls, move.x + 2, move.y)
    : hasWall(state.vWalls, move.x, move.y - 2) || hasWall(state.vWalls, move.x, move.y + 2);
  return `${move.o.toUpperCase()}${sign(net)}${connected ? 'C' : ''}`;
}

const recorded: RecordedGame[] = [];
for (let game = 0; game < games; game++) {
  const rng = rngFrom(seed + game);
  const states: GameState[] = [initialState()];
  const moves: Move[] = [];
  let state = states[0];
  while (!isOver(state)) {
    const move = search(state, {
      maxDepth: depthAt(state.ply, 3, OPENING_PLIES),
      weights: champion.weights,
      margin: 1,
      pickFromTop: pickAt(state.ply, OPENING_PLIES),
      rng,
      recent: recentCells(states, state.turn),
    }).move;
    moves.push(move);
    state = applyMove(state, move);
    states.push(state);
  }
  recorded.push({ states, moves });
}

const observations: Observation[] = [];
outer: for (let game = 0; game < recorded.length; game++) {
  const row = recorded[game];
  const allCoarse = row.moves.map((move, ply) => moveToken(row.states[ply], move, false));
  const allTactical = row.moves.map((move, ply) => moveToken(row.states[ply], move, true));
  for (let ply = game % sampleEvery; ply < row.moves.length; ply += sampleEvery) {
    const state = row.states[ply];
    const choice3 = search(state, { maxDepth: 3, weights: champion.weights, repeatPenalty: 0 }).move;
    const choice4 = search(state, { maxDepth: 4, weights: champion.weights, repeatPenalty: 0 }).move;
    observations.push({
      game: game + 1,
      ply,
      phase: phaseAt(ply),
      changed: moveToString(choice3) !== moveToString(choice4),
      move3: moveToString(choice3),
      move4: moveToString(choice4),
      token3: moveToken(state, choice3, true),
      token4: moveToken(state, choice4, true),
      position: row.moves.slice(0, ply).map(moveToString).join(' '),
      coarse: allCoarse.slice(Math.max(0, ply - context), Math.min(row.moves.length, ply + context)),
      tactical: allTactical.slice(Math.max(0, ply - context), Math.min(row.moves.length, ply + context)),
    });
    if (observations.length >= maxPositions) break outer;
  }
}

type Scope = '直前' | '直後' | '境界';
interface PatternCount { changed: number; stable: number }

function patterns(tokens: string[], before: number, scope: Scope): string[] {
  const found = new Set<string>();
  if (scope === '直前') {
    for (let size = 2; size <= 4 && size <= before; size++) found.add(tokens.slice(before - size, before).join(' '));
  } else if (scope === '直後') {
    for (let size = 2; size <= 4 && before + size <= tokens.length; size++) found.add(tokens.slice(before, before + size).join(' '));
  } else {
    for (let left = 1; left <= 3; left++) {
      for (let right = 1; right <= 3; right++) {
        if (left + right > 4 || left > before || before + right > tokens.length) continue;
        found.add(`${tokens.slice(before - left, before).join(' ')} | ${tokens.slice(before, before + right).join(' ')}`);
      }
    }
  }
  return [...found];
}

function reportPatterns(label: string, key: 'coarse' | 'tactical', scope: Scope, rows: Observation[]) {
  const counts = new Map<string, PatternCount>();
  const changedTotal = rows.filter((row) => row.changed).length;
  const stableTotal = rows.length - changedTotal;
  for (const row of rows) {
    const before = Math.min(context, row.ply);
    for (const pattern of patterns(row[key], before, scope)) {
      const count = counts.get(pattern) ?? { changed: 0, stable: 0 };
      if (row.changed) count.changed++;
      else count.stable++;
      counts.set(pattern, count);
    }
  }
  const ranked = [...counts]
    .filter(([, count]) => count.changed >= 2 && count.changed + count.stable >= minCount)
    .map(([pattern, count]) => ({
      pattern,
      ...count,
      lift: ((count.changed + 1) / (changedTotal + 2)) / ((count.stable + 1) / (stableTotal + 2)),
    }))
    .sort((a, b) => b.lift - a.lift || b.changed - a.changed)
    .slice(0, 8);
  console.log(`\n### ${label}・${scope}`);
  if (ranked.length === 0) console.log('- 条件を満たす並びなし');
  for (const item of ranked) {
    console.log(`- \`${item.pattern}\`: 変更 ${item.changed}/${changedTotal}、安定 ${item.stable}/${stableTotal}、平滑化比 ${item.lift.toFixed(2)}倍`);
  }
}

const changed = observations.filter((row) => row.changed);
const pct = (count: number, total: number) => total === 0 ? '-' : `${(100 * count / total).toFixed(1)}%`;
console.log(`# 第 ${champion.generation} 世代の手順モチーフ分析`);
console.log();
console.log(`収集: ${recorded.length}局から${observations.length}局面（${sampleEvery}手間隔）`);
console.log(`シード: ${seed}`);
console.log(`3手→4手の最善手変更: ${changed.length}/${observations.length}（${pct(changed.length, observations.length)}）`);
console.log('記号: P=駒、H/V=横/縦壁、+=有利な距離変化、0=即時距離差なし、-=不利、C=同方向の壁へ連結');

console.log('\n## フェーズ別');
for (const phase of ['序盤', '中盤', '終盤']) {
  const rows = observations.filter((row) => row.phase === phase);
  const count = rows.filter((row) => row.changed).length;
  console.log(`- ${phase}: ${count}/${rows.length}（${pct(count, rows.length)}）`);
}

console.log('\n## 3手と4手の候補手');
const transitions = new Map<string, number>();
for (const row of changed) {
  const name = `${row.token3} → ${row.token4}`;
  transitions.set(name, (transitions.get(name) ?? 0) + 1);
}
for (const [name, count] of [...transitions].sort((a, b) => b[1] - a[1])) console.log(`- ${name}: ${count}`);

console.log('\n## 変更群に偏る手順');
for (const scope of ['直前', '直後', '境界'] as const) reportPatterns('粗い記号', 'coarse', scope, observations);
for (const scope of ['直前', '直後', '境界'] as const) reportPatterns('効果つき記号', 'tactical', scope, observations);

console.log('\n## 変更局面の例');
for (const row of changed.slice(0, maxExamples)) {
  const before = Math.min(context, row.ply);
  const contextText = [...row.tactical];
  contextText.splice(before, 0, '|');
  console.log(`\n### 対局${row.game}・${row.ply}手目・${row.phase}`);
  console.log(`- 3手: ${row.move3}（${row.token3}） / 4手: ${row.move4}（${row.token4}）`);
  console.log(`- 前後: ${contextText.join(' ')}`);
  console.log(`- 棋譜: ${row.position || '（初期局面）'}`);
}

console.log('\n## 読み方');
console.log('- 同じ対局から複数局面を採る探索的な相関分析であり、平滑化比は独立な因果効果や信頼区間ではない。');
console.log('- 候補は別シードで再現し、境界分析で盤面要素を最小変更し、最後に400局以上の比較で採否を決める。');
