// 3手読みと4手読みの判断境界を、既存壁を1枚だけ除く最小編集で調べる。
// 重い処理なので GitHub Actions の Experiment ワークフローから実行する。
//
// 例: npx tsx scripts/experiments/boundaries.ts --games 30 --positions 400 --per-group 60

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
  cloneState,
  initialState,
  isOver,
  moveToString,
  shortestDistance,
} from '../../src/engine/game';
import { OPENING_PLIES, depthAt, pickAt, rngFrom } from '../../src/train/match';

const { values: args } = parseArgs({
  options: {
    games: { type: 'string', default: '30' },
    positions: { type: 'string', default: '400' },
    'per-group': { type: 'string', default: '60' },
    'sample-every': { type: 'string', default: '4' },
    seed: { type: 'string', default: '93000000' },
    examples: { type: 'string', default: '12' },
  },
});

const champion = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'training', 'champion.json'), 'utf8'));
const games = Number(args.games);
const maxPositions = Number(args.positions);
const perGroup = Number(args['per-group']);
const sampleEvery = Number(args['sample-every']);
const seed = Number(args.seed);
const maxExamples = Number(args.examples);
const other = (p: Player): Player => (1 - p) as Player;

interface WallRecord {
  move: Extract<Move, { type: 'wall' }>;
  owner: Player;
  placedAt: number;
}
interface Position {
  game: number;
  state: GameState;
  moves: Move[];
  walls: WallRecord[];
  move3: Move;
  move4: Move;
  changed: boolean;
}
interface EditResult {
  position: Position;
  wall: WallRecord;
  mode: 'geometry' | 'reserve' | 'both';
  after3: Move;
  after4: Move;
  afterChanged: boolean;
  myBlock: number;
  opBlock: number;
  connected: boolean;
}

function choice(state: GameState, depth: number): Move {
  return search(state, { maxDepth: depth, weights: champion.weights, repeatPenalty: 0 }).move;
}

function isConnected(state: GameState, wall: WallRecord): boolean {
  const { x, y, o } = wall.move;
  const walls = o === 'h' ? state.hWalls : state.vWalls;
  const at = (cx: number, cy: number) => cx >= 0 && cx < WALL_SLOTS && cy >= 0 && cy < WALL_SLOTS && walls[cy * WALL_SLOTS + cx] === 1;
  return o === 'h' ? at(x - 2, y) || at(x + 2, y) : at(x, y - 2) || at(x, y + 2);
}

const positions: Position[] = [];
for (let game = 0; game < games && positions.length < maxPositions; game++) {
  const rng = rngFrom(seed + game);
  const history: GameState[] = [initialState()];
  const moves: Move[] = [];
  const walls: WallRecord[] = [];
  let state = history[0];
  while (!isOver(state) && positions.length < maxPositions) {
    if (state.ply % sampleEvery === game % sampleEvery && state.ply >= 8) {
      const move3 = choice(state, 3);
      const move4 = choice(state, 4);
      positions.push({
        game: game + 1,
        state,
        moves: [...moves],
        walls: [...walls],
        move3,
        move4,
        changed: moveToString(move3) !== moveToString(move4),
      });
    }
    const move = search(state, {
      maxDepth: depthAt(state.ply, 3, OPENING_PLIES),
      weights: champion.weights,
      margin: 1,
      pickFromTop: pickAt(state.ply, OPENING_PLIES),
      rng,
      recent: recentCells(history, state.turn),
    }).move;
    if (move.type === 'wall') walls.push({ move, owner: state.turn, placedAt: state.ply });
    moves.push(move);
    state = applyMove(state, move);
    history.push(state);
  }
}

const changed = positions.filter((row) => row.changed).slice(0, perGroup);
const stable = positions.filter((row) => !row.changed).slice(0, perGroup);
const selected = [...changed, ...stable];
const edits: EditResult[] = [];

for (const position of selected) {
  const me = position.state.turn;
  const beforeMe = shortestDistance(position.state, me);
  const beforeOp = shortestDistance(position.state, other(me));
  for (const wall of position.walls) {
    for (const mode of ['geometry', 'reserve', 'both'] as const) {
      const edited = cloneState(position.state);
      const { x, y, o } = wall.move;
      if (mode !== 'reserve') (o === 'h' ? edited.hWalls : edited.vWalls)[y * WALL_SLOTS + x] = 0;
      if (mode !== 'geometry') edited.wallsLeft[wall.owner]++;
      const after3 = choice(edited, 3);
      const after4 = choice(edited, 4);
      edits.push({
        position,
        wall,
        mode,
        after3,
        after4,
        afterChanged: moveToString(after3) !== moveToString(after4),
        myBlock: beforeMe - shortestDistance(edited, me),
        opBlock: beforeOp - shortestDistance(edited, other(me)),
        connected: isConnected(position.state, wall),
      });
    }
  }
}

const pct = (count: number, total: number) => total === 0 ? '-' : `${(100 * count / total).toFixed(1)}%`;
const avg = (values: number[]) => values.length === 0 ? '-' : (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
const describe = (rows: EditResult[]) => {
  const horizontal = rows.filter((row) => row.wall.move.o === 'h').length;
  const connected = rows.filter((row) => row.connected).length;
  const recent = rows.filter((row) => row.position.state.ply - row.wall.placedAt <= 8).length;
  return `横壁 ${horizontal}/${rows.length}（${pct(horizontal, rows.length)}）、連結壁 ${connected}/${rows.length}（${pct(connected, rows.length)}）、8手以内 ${recent}/${rows.length}（${pct(recent, rows.length)}）、自分を塞ぐ ${avg(rows.map((row) => row.myBlock))}歩、相手を塞ぐ ${avg(rows.map((row) => row.opBlock))}歩`;
};

const both = edits.filter((row) => row.mode === 'both');
const resolved = both.filter((row) => row.position.changed && !row.afterChanged);
const created = both.filter((row) => !row.position.changed && row.afterChanged);
const unchangedFromChanged = edits.filter((row) => row.position.changed && row.afterChanged);
const target = changed.filter((row) => row.move3.type === 'pawn' && row.move4.type === 'wall' && row.move4.o === 'h');
const targetEdits = resolved.filter((row) => target.includes(row.position));
const shallowBecomesDeep = resolved.filter((row) => moveToString(row.after3) === moveToString(row.position.move4));
const deepBecomesShallow = resolved.filter((row) => moveToString(row.after4) === moveToString(row.position.move3));

console.log(`# 第 ${champion.generation} 世代の判断境界分析`);
console.log();
console.log(`収集: ${positions.length}局面、変更群${changed.length}・安定群${stable.length}を編集（シード${seed}）`);
console.log(`壁ごとの編集: 変更群${both.filter((row) => row.position.changed).length}件、安定群${both.filter((row) => !row.position.changed).length}件`);
console.log('\n## 境界をまたいだ編集');
for (const [mode, name] of [['geometry', '盤面から除去だけ'], ['reserve', '残り壁を返すだけ'], ['both', '除去して残り壁も返す']] as const) {
  const modeRows = edits.filter((row) => row.mode === mode);
  const modeResolved = modeRows.filter((row) => row.position.changed && !row.afterChanged);
  const modeCreated = modeRows.filter((row) => !row.position.changed && row.afterChanged);
  console.log(`- ${name}: 解消 ${modeResolved.length}件・${new Set(modeResolved.map((row) => row.position)).size}/${changed.length}局面 / 発生 ${modeCreated.length}件・${new Set(modeCreated.map((row) => row.position)).size}/${stable.length}局面`);
}
console.log(`- 不一致が解消: ${resolved.length}件 / 変更群編集、対象局面 ${new Set(resolved.map((row) => row.position)).size}/${changed.length}`);
console.log(`- 不一致が発生: ${created.length}件 / 安定群編集、対象局面 ${new Set(created.map((row) => row.position)).size}/${stable.length}`);
console.log(`- 解消の向き: 3手側が元の4手へ ${shallowBecomesDeep.length}件 / 4手側が元の3手へ ${deepBecomesShallow.length}件`);
console.log(`- 前進→横壁の対象局面: ${target.length}/${changed.length}、その不一致を解消した編集 ${targetEdits.length}件`);

console.log('\n## 除いた壁の性質');
console.log(`- 不一致を解消: ${describe(resolved)}`);
console.log(`- 不一致を維持: ${describe(unchangedFromChanged.filter((row) => row.mode === 'both'))}`);
console.log(`- 不一致を発生: ${describe(created)}`);
console.log(`- 前進→横壁を解消: ${describe(targetEdits)}`);

const byOrientationAndImpact = new Map<string, { resolved: number; created: number }>();
for (const row of [...resolved, ...created]) {
  const key = `${row.wall.move.o.toUpperCase()} / 自分${row.myBlock}・相手${row.opBlock}`;
  const count = byOrientationAndImpact.get(key) ?? { resolved: 0, created: 0 };
  if (row.position.changed) count.resolved++;
  else count.created++;
  byOrientationAndImpact.set(key, count);
}
console.log('\n## 向きと距離効果');
for (const [name, count] of [...byOrientationAndImpact].sort((a, b) => b[1].resolved + b[1].created - a[1].resolved - a[1].created)) {
  console.log(`- ${name}: 解消${count.resolved} / 発生${count.created}`);
}

console.log('\n## 不一致を解消した例');
for (const row of resolved.slice(0, maxExamples)) {
  const removed = moveToString(row.wall.move);
  console.log(`\n### 対局${row.position.game}・${row.position.state.ply}手目・${removed}を除去`);
  console.log(`- 元: 3手 ${moveToString(row.position.move3)} / 4手 ${moveToString(row.position.move4)}`);
  console.log(`- 編集後: 3手 ${moveToString(row.after3)} / 4手 ${moveToString(row.after4)}`);
  console.log(`- 壁: ${row.wall.move.o.toUpperCase()}、${row.position.state.ply - row.wall.placedAt}手前、自分${row.myBlock}歩・相手${row.opBlock}歩、連結${row.connected ? 'あり' : 'なし'}`);
  console.log(`- 棋譜: ${row.position.moves.map(moveToString).join(' ')}`);
}

console.log('\n## 読み方');
console.log('- 壁の除去は合法性を壊さない最小編集で、置いたプレイヤーへ壁を1枚返して残数も整合させた。');
console.log('- 同じ局面から多数の壁を除く探索的分析なので、編集件数を独立標本数とはみなさない。別シードで局面単位の再現を確認してから特徴量化する。');
