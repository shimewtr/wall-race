import type { Weights } from '../engine/ai';

interface ChampionFile {
  generation: number;
  weights: Partial<Weights>;
}

// 学習済みのチャンピオンがあればビルド時に取り込む（まだ無ければ初期の重みを使う）
const files = import.meta.glob<ChampionFile>('../../training/champion.json', { eager: true, import: 'default' });

export const champion: ChampionFile | null = Object.values(files)[0] ?? null;
