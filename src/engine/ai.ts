// 探索 AI（negamax + αβ枝刈り + 反復深化）。
// 評価関数は重みベクトルで表現しているので、自己対戦でこの重みを調整して強くしていく想定。

import {
  type GameState,
  type Move,
  type Player,
  type Pos,
  DIRS,
  SIZE,
  WALL_SLOTS,
  applyMove,
  canStep,
  cloneState,
  goalDistances,
  makeMove,
  unmakeMove,
  goalRow,
  wallFits,
  isOver,
  pawnMoves,
  shortestDistance,
  shortestPath,
} from './game';

// 評価に使う特徴量。すべて「手番側から見て」大きいほど良い向きで定義する。
// 学習で重みが決まると、そのまま「何を重視すべきか」の人間向けの説明になる。
export const FEATURES = {
  pathDiff: '最短経路の差（相手の残り歩数 - 自分の残り歩数）',
  rowDiff: '壁を無視した前進の差（相手のゴールまでの段数 - 自分の段数）',
  wallDiff: '残り壁の差（自分 - 相手）',
  urgency: '終盤の 1 歩の重さ（1/(自分の歩数+1) - 1/(相手の歩数+1)）',
  wallsWhenClose: '相手がゴールに近いほど自分の壁に価値（自分の壁/(相手の歩数+1) - 相手の壁/(自分の歩数+1)）',
  center: '中央寄り（相手の中央からの横ずれ - 自分の横ずれ）',
  adjacent: '相手と隣接している（手番側は飛び越えのチャンス）',
  lockedRace: '壁が尽きて歩数勝負が確定している（相手の壁 0 で自分が先着できる +1、自分の壁 0 で相手が先着できる -1）',
  wallThreat: '壁 1 枚で相手を遠回りさせられる最大の歩数 − 相手が壁 1 枚で自分を遠回りさせられる最大の歩数（壁が尽きた側は 0）',
  netWallThreat: '壁1枚の相手への遠回りから自分への遠回りを引いた最大値の差（自分 − 相手）',
  wallChain: '壁 2 枚の連携で増やせる追加の遠回り（2 枚の最大効果 − 1 枚の最大効果）の差',
  routeDiversity: '最短ルートの本数の差（自分 − 相手。2 倍ごとに +1、16 本で頭打ち）',
} as const;

export type Feature = keyof typeof FEATURES;
export type Weights = Record<Feature, number>;
export const FEATURE_NAMES = Object.keys(FEATURES) as Feature[];

export const DEFAULT_WEIGHTS: Weights = {
  pathDiff: 10,
  rowDiff: 0,
  wallDiff: 2,
  urgency: 20,
  wallsWhenClose: 0,
  center: 0,
  adjacent: 0,
  lockedRace: 0,
  wallThreat: 0,
  netWallThreat: 0,
  wallChain: 0,
  routeDiversity: 0,
};

// 古い保存データに無い特徴量はデフォルト値で補う
export const withDefaults = (w: Partial<Weights> = {}): Weights => ({ ...DEFAULT_WEIGHTS, ...w });

export interface SearchOptions {
  maxDepth: number;
  // まず maxDepth で読み、壁を選んだ局面だけ指定深さで読み直す（適応深読み実験用）
  wallGateDepth?: number;
  // wallGateDepth を有効にする最初の手数。序盤の展開ばらしを深読みしないために使う
  wallGateAfterPly?: number;
  // 2手と4手が同じ手、3手だけ別の手になる偶奇反復では3手側を採る（実験用）
  evenOscillationFallback?: boolean;
  timeMs?: number;
  weights?: Partial<Weights>;
  // 評価値に加えるランダムノイズの幅。弱い手も選ばれうるので、わざと手加減するレベル向け
  noise?: number;
  // 最善手からこの点数差以内の手の中からランダムに選ぶ（1 歩 = 10 点）。
  // 強さを落とさずに毎回同じ棋譜になるのを避けるためのもの
  margin?: number;
  rng?: () => number;
  // 自分のコマが直近にいたマス。ここへ戻る手は少し減点して、行ったり来たりを防ぐ
  recent?: Pos[];
  // 戻る手の減点（1 歩 = 10 点。同点に近い手どうしの順位だけを変える大きさ）
  repeatPenalty?: number;
  // いま指す手の候補に、守りの壁やコマ周りの壁も含める（読みの途中は相手の経路をふさぐ壁だけ）
  wideRoot?: boolean;
  // 評価の上位この数の手からランダムに選ぶ（自己対戦の序盤をばらけさせる用）。2 以上なら margin より優先
  pickFromTop?: number;
}

export const REPEAT_PENALTY = 8;
// 根でも相手の最短経路をふさぐ壁だけを候補にする。
// 広い候補との比較は 3 手・4 手読みとも 400 局で互角、狭い候補は約 4〜7 倍速かった。
export const WIDE_ROOT_DEFAULT = false;

// 対局の状態履歴から、player のコマが直近 n 手番でいたマスを返す（現在地は除く）
export function recentCells(history: GameState[], player: Player, n = 4): Pos[] {
  const cur = history[history.length - 1].pawns[player];
  const cells: Pos[] = [];
  for (let i = history.length - 2; i >= 0 && cells.length < n; i--) {
    if (history[i].turn !== player) continue;
    const p = history[i].pawns[player];
    if (p.x !== cur.x || p.y !== cur.y) cells.push(p);
  }
  return cells;
}

export interface SearchResult {
  move: Move;
  score: number;
  // 最終反復での根の1位と2位の評価差。適応深読みの分析用で、手の選択には使わない。
  rootGap?: number;
  depth: number;
  nodes: number;
}

const sameMove = (a: Move, b: Move) =>
  a.type === b.type && a.x === b.x && a.y === b.y &&
  (a.type === 'pawn' || (b.type === 'wall' && a.o === b.o));

// 勝ちを読み切ったときの評価値（実際は WIN − 勝つまでの手数）
export const WIN = 100000;

class Timeout extends Error {}

// maxDetour の結果のキャッシュ。読みの途中では「壁の配置と相手のコマの位置」が同じ局面が何度も出てくる。
// 壁を置く側のコマの位置は「自分の道までふさぐ壁は置けない」の判定にしか効かず、結果が変わることはまれなので
// キーに含めない（含めると自分のコマが動くたびに計算し直しになる）。
// キーは 32 ビットのハッシュなので衝突はありうるが、評価の一項目の近似なので許容する
interface DetourResult {
  value: number;
  // 即時効果が同率最大の壁をすべて残す。wallChain が走査順で1枚を選ぶと、
  // 左右反転した等価局面で別の2枚目を評価してしまうため。
  walls: Array<{ x: number; y: number; o: 'h' | 'v'; afterDist: number }>;
}
const detourCache = new Map<number, DetourResult>();
const netDetourCache = new Map<number, number>();
const wallChainCache = new Map<number, number>();
// 計測用（profile 実験で使う）
export const detourStats = { calls: 0, misses: 0 };
const DETOUR_CACHE_LIMIT = 500_000;

function detourKey(s: GameState, attacker: Player): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.hWalls.length; i++) {
    if (s.hWalls[i]) h = Math.imul(h ^ (i + 1), 0x01000193);
    if (s.vWalls[i]) h = Math.imul(h ^ (i + 101), 0x01000193);
  }
  const v = s.pawns[1 - attacker];
  return Math.imul(h ^ (attacker * 81 + v.y * 9 + v.x), 0x01000193) >>> 0;
}

function netDetourKey(s: GameState, attacker: Player): number {
  const a = s.pawns[attacker];
  return Math.imul(detourKey(s, attacker) ^ (201 + a.y * SIZE + a.x), 0x01000193) >>> 0;
}

// attacker が壁 1 枚で相手（victim）の最短歩数を最大何歩伸ばせるか。壁が尽きていれば 0
export function maxDetour(s: GameState, attacker: Player, victimDist?: number): number {
  if (s.wallsLeft[attacker] <= 0) return 0;
  detourStats.calls++;
  const key = detourKey(s, attacker);
  const cached = detourCache.get(key);
  if (cached !== undefined) return cached.value;
  detourStats.misses++;
  const result = computeBestDetour(s, attacker, victimDist ?? shortestDistance(s, (1 - attacker) as Player));
  if (detourCache.size >= DETOUR_CACHE_LIMIT) detourCache.clear();
  detourCache.set(key, result);
  return result.value;
}

// 壁1枚による相手の遠回りから、自分の遠回りを引いた正味の最大効果。
// 同じ走査で通常の maxDetour も計算・保存し、両特徴を使うときの二重計算を避ける。
export function maxNetDetour(
  s: GameState,
  attacker: Player,
  victimDist = shortestDistance(s, (1 - attacker) as Player),
  attackerDist = shortestDistance(s, attacker),
): number {
  if (s.wallsLeft[attacker] <= 0) return 0;
  const key = netDetourKey(s, attacker);
  const cached = netDetourCache.get(key);
  if (cached !== undefined) return cached;

  const victim = (1 - attacker) as Player;
  const victimPath = shortestPath(s, victim);
  const attackerPath = shortestPath(s, attacker);
  if (!victimPath || !attackerPath) return 0;
  const seen = new Set<number>();
  let best = 0;
  let rawBest = 0;
  let rawWalls: DetourResult['walls'] = [];
  const tryWall = (x: number, y: number, o: 'h' | 'v') => {
    if (x < 0 || y < 0 || x >= WALL_SLOTS || y >= WALL_SLOTS || !wallFits(s, x, y, o)) return;
    const wallKey = (o === 'h' ? 0 : WALL_SLOTS * WALL_SLOTS) + y * WALL_SLOTS + x;
    if (seen.has(wallKey)) return;
    seen.add(wallKey);
    const walls = o === 'h' ? s.hWalls : s.vWalls;
    const i = y * WALL_SLOTS + x;
    walls[i] = 1;
    const victimAfter = shortestDistance(s, victim);
    const raw = victimAfter < 0 ? 0 : Math.max(0, victimAfter - victimDist);
    // 自分への遠回りは 0 以上なので、raw が現在値を超えない壁は正味でも更新できない。
    // その場合は自分側の経路探索を省く。
    const attackerAfter =
      raw > best && cutsPath(attackerPath, x, y, o) ? shortestDistance(s, attacker) : attackerDist;
    walls[i] = 0;
    if (victimAfter < 0) return;
    if (rawWalls.length === 0 || raw > rawBest) {
      rawBest = raw;
      rawWalls = [{ x, y, o, afterDist: victimAfter }];
    } else if (raw === rawBest) {
      rawWalls.push({ x, y, o, afterDist: victimAfter });
    }
    if (raw > best && attackerAfter >= 0) {
      best = Math.max(best, raw - Math.max(0, attackerAfter - attackerDist));
    }
  };
  for (let k = 0; k + 1 < victimPath.length; k++) {
    const a = victimPath[k];
    const b = victimPath[k + 1];
    if (a.x === b.x) {
      const y = Math.min(a.y, b.y);
      tryWall(a.x, y, 'h');
      tryWall(a.x - 1, y, 'h');
    } else {
      const x = Math.min(a.x, b.x);
      tryWall(x, a.y, 'v');
      tryWall(x, a.y - 1, 'v');
    }
  }
  const rawKey = detourKey(s, attacker);
  if (detourCache.size >= DETOUR_CACHE_LIMIT) detourCache.clear();
  detourCache.set(rawKey, { value: rawBest, walls: rawWalls });
  if (netDetourCache.size >= DETOUR_CACHE_LIMIT) netDetourCache.clear();
  netDetourCache.set(key, best);
  return best;
}

// 壁を置く側の道までふさいでしまう壁かどうかは確かめない（そうなる壁はまれで、評価の一項目の近似としては許容する。
// 確かめると 1 回ごとに経路探索が 1 回増える）。相手の道を完全にふさぐ壁は置けないので除く
function computeBestDetour(s: GameState, attacker: Player, victimDist: number): DetourResult {
  const victim = (1 - attacker) as Player;
  const path = shortestPath(s, victim);
  if (!path) return { value: 0, walls: [] };
  const seen = new Set<number>();
  let best = 0;
  let bestWalls: DetourResult['walls'] = [];
  const tryWall = (x: number, y: number, o: 'h' | 'v') => {
    if (x < 0 || y < 0 || x >= WALL_SLOTS || y >= WALL_SLOTS || !wallFits(s, x, y, o)) return;
    const wallKey = (o === 'h' ? 0 : WALL_SLOTS * WALL_SLOTS) + y * WALL_SLOTS + x;
    if (seen.has(wallKey)) return;
    seen.add(wallKey);
    const walls = o === 'h' ? s.hWalls : s.vWalls;
    const i = y * WALL_SLOTS + x;
    walls[i] = 1;
    const d = shortestDistance(s, victim);
    walls[i] = 0;
    if (d < 0) return;
    const gain = Math.max(0, d - victimDist);
    if (bestWalls.length === 0 || gain > best) {
      best = gain;
      bestWalls = [{ x, y, o, afterDist: d }];
    } else if (gain === best) {
      bestWalls.push({ x, y, o, afterDist: d });
    }
  };
  for (let k = 0; k + 1 < path.length; k++) {
    const a = path[k];
    const b = path[k + 1];
    if (a.x === b.x) {
      const y = Math.min(a.y, b.y);
      tryWall(a.x, y, 'h');
      tryWall(a.x - 1, y, 'h');
    } else {
      const x = Math.min(a.x, b.x);
      tryWall(x, a.y, 'v');
      tryWall(x, a.y - 1, 'v');
    }
  }
  return { value: best, walls: bestWalls };
}

// 2 枚の壁を続けて使ったとき、最善の 1 枚だけより何歩多く遠回りさせられるか。
// 左右反転と「上下反転＋プレイヤー交換」で同じ基準になるよう、攻撃側から見た盤面を正規化する。
// 左右非対称な最初の要素を左側へ寄せる向きを canonical とする。
function canonicalFlipX(s: GameState, attacker: Player): boolean {
  const victim = (1 - attacker) as Player;
  for (const x of [s.pawns[attacker].x, s.pawns[victim].x]) {
    const mirrored = SIZE - 1 - x;
    if (x !== mirrored) return mirrored < x;
  }
  for (let normalizedY = 0; normalizedY < WALL_SLOTS; normalizedY++) {
    const y = attacker === 0 ? normalizedY : WALL_SLOTS - 1 - normalizedY;
    for (const walls of [s.hWalls, s.vWalls]) {
      for (let x = 0; x < WALL_SLOTS; x++) {
        const normal = walls[y * WALL_SLOTS + x];
        const mirrored = walls[y * WALL_SLOTS + WALL_SLOTS - 1 - x];
        if (normal !== mirrored) return mirrored < normal;
      }
    }
  }
  return false;
}

function canonicalDetourWall(s: GameState, attacker: Player, walls: DetourResult['walls']) {
  const flipX = canonicalFlipX(s, attacker);
  const key = (wall: DetourResult['walls'][number]) => {
    const y = attacker === 0 ? wall.y : WALL_SLOTS - 1 - wall.y;
    const x = flipX ? WALL_SLOTS - 1 - wall.x : wall.x;
    return (y * WALL_SLOTS + x) * 2 + (wall.o === 'h' ? 0 : 1);
  };
  let best = walls[0];
  let bestKey = key(best);
  for (let i = 1; i < walls.length; i++) {
    const candidateKey = key(walls[i]);
    if (candidateKey < bestKey) {
      best = walls[i];
      bestKey = candidateKey;
    }
  }
  return best;
}

// 全候補を二重に調べると評価が重くなりすぎるため、1 枚目は即時効果が最大の候補に絞る。
// 同率なら盤面を正規化した順で1枚だけ選び、対称性と速度を両立する。
export function maxWallChain(s: GameState, attacker: Player, victimDist?: number): number {
  if (s.wallsLeft[attacker] < 2) return 0;
  const key = detourKey(s, attacker);
  const cached = wallChainCache.get(key);
  if (cached !== undefined) return cached;

  const victim = (1 - attacker) as Player;
  const baseDist = victimDist ?? shortestDistance(s, victim);
  let first = detourCache.get(key);
  if (!first) {
    first = computeBestDetour(s, attacker, baseDist);
    if (detourCache.size >= DETOUR_CACHE_LIMIT) detourCache.clear();
    detourCache.set(key, first);
  }
  if (first.walls.length === 0) return 0;
  const c = canonicalDetourWall(s, attacker, first.walls);
  const walls = c.o === 'h' ? s.hWalls : s.vWalls;
  const i = c.y * WALL_SLOTS + c.x;
  walls[i] = 1;
  // 1枚目の効果は maxDetour と共通なので、追加分は置いた後に残る「次の1枚」の効果そのもの。
  // この「1枚目を置いた盤面」は実際の探索でも現れうるので、2枚目の計算結果も通常キャッシュへ残す。
  const result = maxDetour(s, attacker, c.afterDist);
  walls[i] = 0;
  if (wallChainCache.size >= DETOUR_CACHE_LIMIT) wallChainCache.clear();
  wallChainCache.set(key, result);
  return result;
}

// 最短ルートの本数を 16 本まで数え、log2 にした値（1 本=0、2 本=1、4 本=2、…）。
// 本数をそのまま使うと開けた局面の巨大な値に引っ張られるため、対数かつ上限付きにする。
const ROUTE_COUNTS = new Uint8Array(SIZE * SIZE);
const ROUTE_COUNT_CAP = 16;
const routeCache = new Map<number, number>();
const ROUTE_CACHE_LIMIT = 500_000;

function routeKey(s: GameState, p: Player): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.hWalls.length; i++) {
    if (s.hWalls[i]) h = Math.imul(h ^ (i + 1), 0x01000193);
    if (s.vWalls[i]) h = Math.imul(h ^ (i + 101), 0x01000193);
  }
  const pawn = s.pawns[p];
  return Math.imul(h ^ (p * 81 + pawn.y * 9 + pawn.x), 0x01000193) >>> 0;
}

function computeShortestRouteDiversity(s: GameState, p: Player): number {
  const distances = goalDistances(s, p);
  const start = s.pawns[p];
  const startIdx = start.y * SIZE + start.x;
  const startDist = distances[startIdx];
  if (startDist <= 0) return 0;

  ROUTE_COUNTS.fill(0);
  const goal = goalRow(p);
  for (let x = 0; x < SIZE; x++) ROUTE_COUNTS[goal * SIZE + x] = 1;
  for (let d = 1; d <= startDist; d++) {
    for (let idx = 0; idx < distances.length; idx++) {
      if (distances[idx] !== d) continue;
      const x = idx % SIZE;
      const y = (idx - x) / SIZE;
      let count = 0;
      for (const [dx, dy] of DIRS) {
        if (!canStep(s, x, y, dx, dy)) continue;
        const next = (y + dy) * SIZE + x + dx;
        if (distances[next] === d - 1) count += ROUTE_COUNTS[next];
      }
      ROUTE_COUNTS[idx] = Math.min(count, ROUTE_COUNT_CAP);
    }
  }
  return Math.log2(ROUTE_COUNTS[startIdx] || 1);
}

export function shortestRouteDiversity(s: GameState, p: Player): number {
  const key = routeKey(s, p);
  const cached = routeCache.get(key);
  if (cached !== undefined) return cached;
  const result = computeShortestRouteDiversity(s, p);
  if (routeCache.size >= ROUTE_CACHE_LIMIT) routeCache.clear();
  routeCache.set(key, result);
  return result;
}

// myDist / opDist（手番側・相手の最短歩数）が分かっていれば渡すと探索を省ける。
// withThreat が false なら重い wallThreat は計算せず 0 にする
export function features(
  s: GameState,
  myDist = shortestDistance(s, s.turn),
  opDist = shortestDistance(s, (1 - s.turn) as Player),
  withThreat = true,
  withRouteDiversity = true,
  withWallChain = true,
  withNetWallThreat = true,
): Weights {
  const me = s.turn;
  const op = (1 - me) as Player;
  const my = s.pawns[me];
  const o = s.pawns[op];
  // 正味効果の計算は通常の壁脅威キャッシュも埋めるので先に行う。
  const myNetThreat = withNetWallThreat ? maxNetDetour(s, me, opDist, myDist) : 0;
  const opNetThreat = withNetWallThreat ? maxNetDetour(s, op, myDist, opDist) : 0;
  return {
    pathDiff: opDist - myDist,
    rowDiff: Math.abs(goalRow(op) - o.y) - Math.abs(goalRow(me) - my.y),
    wallDiff: s.wallsLeft[me] - s.wallsLeft[op],
    urgency: 1 / (myDist + 1) - 1 / (opDist + 1),
    wallsWhenClose: s.wallsLeft[me] / (opDist + 1) - s.wallsLeft[op] / (myDist + 1),
    center: Math.abs(o.x - 4) - Math.abs(my.x - 4),
    adjacent: Math.abs(my.x - o.x) + Math.abs(my.y - o.y) === 1 ? 1 : 0,
    // 手番側は同じ歩数なら先に着く。壁が無ければ相手は邪魔できない
    lockedRace:
      (s.wallsLeft[op] === 0 && myDist <= opDist ? 1 : 0) - (s.wallsLeft[me] === 0 && opDist < myDist ? 1 : 0),
    wallThreat: withThreat ? maxDetour(s, me, opDist) - maxDetour(s, op, myDist) : 0,
    netWallThreat: myNetThreat - opNetThreat,
    wallChain: withWallChain
      ? maxWallChain(s, me, opDist) - maxWallChain(s, op, myDist)
      : 0,
    routeDiversity: withRouteDiversity ? shortestRouteDiversity(s, me) - shortestRouteDiversity(s, op) : 0,
  };
}

// 手番側から見た静的評価
export function evaluate(
  s: GameState,
  w: Weights = DEFAULT_WEIGHTS,
  myDist?: number,
  opDist?: number,
): number {
  if (s.winner !== null) return s.winner === s.turn ? WIN : -WIN;
  if (s.draw) return 0;
  const f = features(
    s,
    myDist,
    opDist,
    w.wallThreat !== 0,
    w.routeDiversity !== 0,
    w.wallChain !== 0,
    w.netWallThreat !== 0,
  );
  let score = 0;
  for (const k of FEATURE_NAMES) score += w[k] * f[k];
  return score;
}

// 壁 (x, y, o) が経路のどこかの一歩をふさぐか
function cutsPath(path: Pos[], x: number, y: number, o: 'h' | 'v'): boolean {
  for (let i = 0; i + 1 < path.length; i++) {
    const a = path[i];
    const b = path[i + 1];
    if (a.x === b.x) {
      if (o === 'h' && y === Math.min(a.y, b.y) && (x === a.x || x === a.x - 1)) return true;
    } else if (o === 'v' && x === Math.min(a.x, b.x) && (y === a.y || y === a.y - 1)) {
      return true;
    }
  }
  return false;
}

// 探索する壁の候補。
// 基本は「相手の最短経路の一歩一歩をふさぐ壁」だけ（読みの途中はこれで十分かつ軽い）。
// wide のときは加えて、
//   - どちらかのコマに接する壁（足止め・回り込みの準備）
//   - 自分の最短経路に接していて、自分の経路は伸ばさない壁（相手に道をふさがれないための守りの壁）
// も候補にする。根（いま指す手）でだけ使う。
export function candidateWalls(s: GameState, wide = false): Move[] {
  return wallCandidates(s, wide).map((c) => c.m);
}

// 壁を置いた後の歩数（-1 は「その人の最短経路を横切らないので変わらない」）も一緒に返す。探索で距離を引き継ぐのに使う
interface WallCandidate {
  m: Move;
  opAfter: number;
  meAfter: number;
}

function wallCandidates(s: GameState, wide: boolean): WallCandidate[] {
  if (s.wallsLeft[s.turn] <= 0) return [];
  const me = s.turn;
  const op = (1 - me) as Player;
  const path = shortestPath(s, op);
  if (!path) return [];
  const myPath = shortestPath(s, me);
  const seen = new Set<number>();
  const result: WallCandidate[] = [];
  // 合法判定（isLegalWall と同じ結果）。今の最短経路を横切らない壁ではその人の道はふさがらないので、
  // 経路を横切るときだけ探索し直す。不合法なら null
  const legal = (x: number, y: number, o: 'h' | 'v'): WallCandidate | null => {
    if (!wallFits(s, x, y, o)) return null;
    const m: Move = { type: 'wall', x, y, o };
    const cutsOp = cutsPath(path, x, y, o);
    const cutsMe = myPath !== null && cutsPath(myPath, x, y, o);
    if (!cutsOp && !cutsMe) return { m, opAfter: -1, meAfter: -1 };
    const walls = o === 'h' ? s.hWalls : s.vWalls;
    const i = y * WALL_SLOTS + x;
    walls[i] = 1;
    const opAfter = cutsOp ? shortestDistance(s, op) : -1;
    const meAfter = cutsMe ? shortestDistance(s, me) : -1;
    walls[i] = 0;
    if ((cutsOp && opAfter < 0) || (cutsMe && meAfter < 0)) return null;
    return { m, opAfter, meAfter };
  };
  const add = (x: number, y: number, o: 'h' | 'v', keep: (m: Move) => boolean = () => true) => {
    if (x < 0 || y < 0 || x >= WALL_SLOTS || y >= WALL_SLOTS) return;
    const key = (o === 'h' ? 0 : 64) + y * WALL_SLOTS + x;
    if (seen.has(key)) return;
    seen.add(key);
    const c = legal(x, y, o);
    if (c && keep(c.m)) result.push(c);
  };
  // マス (cx, cy) の辺に接する壁のアンカー
  const around = (cx: number, cy: number, keep?: (m: Move) => boolean) => {
    for (const ay of [cy - 1, cy]) {
      for (const ax of [cx - 1, cx]) {
        add(ax, ay, 'h', keep);
        add(ax, ay, 'v', keep);
      }
    }
  };
  for (let i = 0; i + 1 < path.length; i++) {
    const a = path[i];
    const b = path[i + 1];
    if (a.x === b.x) {
      const y = Math.min(a.y, b.y);
      add(a.x, y, 'h');
      add(a.x - 1, y, 'h');
    } else {
      const x = Math.min(a.x, b.x);
      add(x, a.y, 'v');
      add(x, a.y - 1, 'v');
    }
  }
  if (!wide) return result;
  for (const p of s.pawns) around(p.x, p.y);
  if (myPath) {
    const myDist = myPath.length - 1;
    const keepsMyPath = (m: Move) => shortestDistance(applyMove(s, m), me) === myDist;
    for (const c of myPath) around(c.x, c.y, keepsMyPath);
  }
  return result;
}

// 探索で指す手。d は前進した後の自分の歩数、opAfter / meAfter は壁を置いた後の歩数（-1 は変わらない）
interface Candidate {
  m: Move;
  d: number;
  opAfter: number;
  meAfter: number;
}

function orderedMoves(s: GameState, wide = false): Candidate[] {
  // 前進の手は「移動先からゴールまでの歩数」が小さい順。距離はゴール側から 1 回で全マス分求める
  const dist = goalDistances(s, s.turn);
  const pawns: Candidate[] = pawnMoves(s).map((p) => {
    const d = dist[p.y * SIZE + p.x];
    return { m: { type: 'pawn', x: p.x, y: p.y }, d: d < 0 ? Infinity : d, opAfter: -1, meAfter: -1 };
  });
  pawns.sort((a, b) => a.d - b.d);
  return [...pawns, ...wallCandidates(s, wide).map((c) => ({ ...c, d: -1 }))];
}

// 手を指した後の局面の [手番側の歩数, 相手の歩数]（手番が入れ替わるので myDist / opDist も入れ替わる）
function childDistances(c: Candidate, myDist: number, opDist: number): [number, number] {
  if (c.m.type === 'pawn') return [opDist, c.d];
  return [c.opAfter < 0 ? opDist : c.opAfter, c.meAfter < 0 ? myDist : c.meAfter];
}

export function search(s: GameState, opts: SearchOptions): SearchResult {
  const gateDepth = opts.wallGateDepth ?? 0;
  if (gateDepth > opts.maxDepth && s.ply >= (opts.wallGateAfterPly ?? 0)) {
    const base = search(s, { ...opts, wallGateDepth: undefined, wallGateAfterPly: undefined });
    if (base.move.type !== 'wall') return base;
    const deep = search(s, {
      ...opts,
      maxDepth: gateDepth,
      wallGateDepth: undefined,
      wallGateAfterPly: undefined,
    });
    return { ...deep, nodes: base.nodes + deep.nodes };
  }
  const weights = withDefaults(opts.weights);
  const rng = opts.rng ?? Math.random;
  const noise = opts.noise ?? 0;
  const margin = opts.margin ?? 0;
  const pickFromTop = opts.pickFromTop ?? 1;
  const deadline = opts.timeMs ? Date.now() + opts.timeMs : Infinity;
  let nodes = 0;

  const negamax = (
    st: GameState,
    depth: number,
    alpha: number,
    beta: number,
    ply: number,
    myDist: number,
    opDist: number,
  ): number => {
    nodes++;
    if ((nodes & 255) === 0 && Date.now() > deadline) throw new Timeout();
    if (isOver(st)) {
      // 早く勝つ・遅く負けるほうを好むように ply を加味
      if (st.winner === null) return 0;
      return st.winner === st.turn ? WIN - ply : -WIN + ply;
    }
    if (depth === 0) return evaluate(st, weights, myDist, opDist);
    let best = -Infinity;
    for (const c of orderedMoves(st)) {
      const [cm, co] = childDistances(c, myDist, opDist);
      const undo = makeMove(st, c.m);
      const v = -negamax(st, depth - 1, -beta, -alpha, ply + 1, cm, co);
      unmakeMove(st, c.m, undo);
      if (v > best) best = v;
      if (v > alpha) alpha = v;
      if (alpha >= beta) break;
    }
    return best === -Infinity ? evaluate(st, weights, myDist, opDist) : best;
  };

  const rootMoves = orderedMoves(s, opts.wideRoot ?? WIDE_ROOT_DEFAULT);
  const rootMy = shortestDistance(s, s.turn);
  const rootOp = shortestDistance(s, (1 - s.turn) as Player);
  const recent = opts.recent ?? [];
  const repeatPenalty = opts.repeatPenalty ?? REPEAT_PENALTY;
  const penalty = (m: Move) =>
    m.type === 'pawn' && recent.some((p) => p.x === m.x && p.y === m.y) ? repeatPenalty : 0;
  if (rootMoves.length === 0) throw new Error('no legal moves');
  let result: SearchResult = { move: rootMoves[0].m, score: 0, depth: 0, nodes: 0 };
  const iterations: SearchResult[] = [];

  for (let depth = 1; depth <= opts.maxDepth; depth++) {
    // 時間切れで途中で抜けると盤面が戻らないので、深さごとに作業用の複製を使う
    const work = cloneState(s);
    try {
      let bestScore = -Infinity;
      let bestMove = rootMoves[0].m;
      let alpha = -Infinity;
      const scored: Array<{ c: Candidate; v: number }> = [];
      for (const c of rootMoves) {
        const [cm, co] = childDistances(c, rootMy, rootOp);
        // 最善手から margin 以内の手も正確な値が出るよう、窓を margin 分広げて読む。
        // 上位から選ぶときはすべての手の正確な値が要るので、根では枝刈りしない
        const floor = pickFromTop > 1 ? -Infinity : alpha - margin;
        const undo = makeMove(work, c.m);
        const v =
          -negamax(work, depth - 1, -Infinity, -floor, 1, cm, co) -
          penalty(c.m) +
          (noise ? (rng() - 0.5) * noise : 0);
        unmakeMove(work, c.m, undo);
        scored.push({ c, v });
        if (v > bestScore) {
          bestScore = v;
          bestMove = c.m;
        }
        if (v > alpha) alpha = v;
      }
      if (pickFromTop > 1) {
        const top = [...scored].sort((a, b) => b.v - a.v).slice(0, pickFromTop);
        bestMove = top[Math.floor(rng() * top.length)].c.m;
      } else if (margin > 0) {
        const near = scored.filter((x) => x.v >= bestScore - margin);
        bestMove = near[Math.floor(rng() * near.length)].c.m;
      }
      const rankedScores = scored.map((x) => x.v).sort((a, b) => b - a);
      const rootGap = rankedScores.length > 1 ? rankedScores[0] - rankedScores[1] : Infinity;
      result = { move: bestMove, score: bestScore, rootGap, depth, nodes };
      iterations.push(result);
      // 次の深さでは良かった手から読む
      scored.sort((a, b) => b.v - a.v);
      rootMoves.splice(0, rootMoves.length, ...scored.map((x) => x.c));
      if (Math.abs(bestScore) > WIN / 2) break;
    } catch (e) {
      if (e instanceof Timeout) break;
      throw e;
    }
  }
  if (opts.evenOscillationFallback) {
    const depth2 = iterations.find((row) => row.depth === 2);
    const depth3 = iterations.find((row) => row.depth === 3);
    const depth4 = iterations.find((row) => row.depth === 4);
    if (depth2 && depth3 && depth4 && sameMove(depth2.move, depth4.move) && !sameMove(depth3.move, depth4.move)) {
      result = { ...depth3, depth: depth4.depth, nodes };
    }
  }
  result.nodes = nodes;
  return result;
}

export const LEVELS = {
  easy: { maxDepth: 1, noise: 8 },
  normal: { maxDepth: 2, noise: 2 },
  // 第 7 世代では 3 手読みに 70.5%（400 局）。狭い根候補の 4 手読みは平均 13ms / 最大 44ms。
  hard: { maxDepth: 4, timeMs: 5000, margin: 1 },
} satisfies Record<string, SearchOptions>;

export type Level = keyof typeof LEVELS;
