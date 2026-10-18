// AI 同士を対戦させて勝率を出す。自己対戦・パラメータ調整の土台。
// 例: npm run arena -- --games 20 --a '{"maxDepth":2}' --b '{"maxDepth":1}'

import { parseArgs } from 'node:util';
import { type SearchOptions, search } from '../src/engine/ai';
import { applyMoveInPlace, initialState, isOver, moveToString } from '../src/engine/game';

const { values } = parseArgs({
  options: {
    games: { type: 'string', default: '10' },
    a: { type: 'string', default: '{"maxDepth":2,"noise":2}' },
    b: { type: 'string', default: '{"maxDepth":1,"noise":2}' },
    verbose: { type: 'boolean', default: false },
  },
});

const configs: [SearchOptions, SearchOptions] = [JSON.parse(values.a!), JSON.parse(values.b!)];
const games = Number(values.games);
const score = { a: 0, b: 0, draw: 0 };

for (let g = 0; g < games; g++) {
  // 先手後手を交互に入れ替える
  const aIsFirst = g % 2 === 0;
  const s = initialState();
  const record: string[] = [];
  while (!isOver(s)) {
    const cfg = (s.turn === 0) === aIsFirst ? configs[0] : configs[1];
    const { move } = search(s, cfg);
    record.push(moveToString(move));
    applyMoveInPlace(s, move);
  }
  if (s.winner === null) score.draw++;
  else if ((s.winner === 0) === aIsFirst) score.a++;
  else score.b++;
  console.log(`game ${g + 1}: ${s.winner === null ? 'draw' : (s.winner === 0) === aIsFirst ? 'A' : 'B'} (${record.length} plies)`);
  if (values.verbose) console.log(record.join(' '));
}

console.log(`\nA ${score.a} - B ${score.b} (draw ${score.draw})  A win rate ${(score.a / games * 100).toFixed(1)}%`);
