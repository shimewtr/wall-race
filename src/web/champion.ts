import type { Weights } from '../engine/ai';
import type { Calibration } from '../train/calibrate';

interface ChampionFile {
  generation: number;
  weights: Partial<Weights>;
}

// 学習済みのチャンピオンがあればビルド時に取り込む（まだ無ければ初期の重みを使う）
const files = import.meta.glob<ChampionFile>('../../training/champion.json', { eager: true, import: 'default' });

export const champion: ChampionFile | null = Object.values(files)[0] ?? null;

// 評価値を勝率に換算する式（学習で昇格したときに作り直される）
const calibrationFiles = import.meta.glob<Calibration>('../../training/calibration.json', { eager: true, import: 'default' });
const cal = Object.values(calibrationFiles)[0] ?? null;

// チャンピオンと同じ世代の式だけを使う（世代が違うと評価値の尺度が違う）
export const calibration: Calibration | null = cal && champion && cal.generation === champion.generation ? cal : null;
