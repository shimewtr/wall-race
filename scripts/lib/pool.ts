// 対局などの重い処理をワーカースレッドに分散して全コアで回す。

import { availableParallelism } from 'node:os';
import { Worker } from 'node:worker_threads';
import type { Weights } from '../../src/engine/ai';
import type { MatchConfig } from '../../src/train/match';
import type { GameRecord, RecordJob } from '../../src/train/record';
import type { GuideJob } from '../../src/guide/match';
import type { DiscoveryGame, DiscoveryJob } from '../../src/guide/discover';
import type { CounterfactualJob, CounterfactualResult } from '../../src/guide/counterfactual';

export interface PairJob {
  a: Partial<Weights>;
  b: Partial<Weights>;
  opts: Omit<MatchConfig, 'weights'>;
}

// file のワーカーに J を渡して R を受け取る
export class WorkerPool<J, R> {
  private workers: Worker[] = [];
  private idle: Worker[] = [];
  private queue: Array<{ job: J; resolve: (v: R) => void; reject: (e: unknown) => void }> = [];

  constructor(file: URL, size = availableParallelism()) {
    for (let i = 0; i < size; i++) {
      // tsx の環境を引き継いで TypeScript のままワーカーを起動する
      const w = new Worker(file, { execArgv: ['--import', 'tsx'] });
      this.workers.push(w);
      this.idle.push(w);
    }
  }

  get size(): number {
    return this.workers.length;
  }

  run(job: J): Promise<R> {
    return new Promise((resolve, reject) => {
      this.queue.push({ job, resolve, reject });
      this.pump();
    });
  }

  runAll(jobs: J[]): Promise<R[]> {
    return Promise.all(jobs.map((j) => this.run(j)));
  }

  private pump() {
    while (this.idle.length > 0 && this.queue.length > 0) {
      const w = this.idle.pop()!;
      const { job, resolve, reject } = this.queue.shift()!;
      const done = () => {
        w.off('message', onMessage);
        w.off('error', onError);
        this.idle.push(w);
        this.pump();
      };
      const onMessage = (v: R) => {
        done();
        resolve(v);
      };
      const onError = (e: unknown) => {
        done();
        reject(e);
      };
      w.on('message', onMessage);
      w.on('error', onError);
      w.postMessage(job);
    }
  }

  async close(): Promise<void> {
    await Promise.all(this.workers.map((w) => w.terminate()));
  }
}

// 2 つの重みを先後入れ替えで対戦させ、a の得点（0〜2）を返す
export class PairPool extends WorkerPool<PairJob, number> {
  constructor(size?: number) {
    super(new URL('./pool-worker.ts', import.meta.url), size);
  }
}

// 1 局を記録付きで指す（手引き作りの分析用）
export class RecordPool extends WorkerPool<RecordJob, GameRecord> {
  constructor(size?: number) {
    super(new URL('./record-worker.ts', import.meta.url), size);
  }
}

// 手引き AI と探索 AI の 1 局（手引き AI の得点を返す）
export class GuidePool extends WorkerPool<GuideJob, number> {
  constructor(size?: number) {
    super(new URL('./guide-worker.ts', import.meta.url), size);
  }
}

// 手引き AI と探索 AI が違う手を選ぶ局面を記録する（戦術発見用）
export class DiscoveryPool extends WorkerPool<DiscoveryJob, DiscoveryGame> {
  constructor(size?: number) {
    super(new URL('./discover-worker.ts', import.meta.url), size);
  }
}

// 発見局面で 2 つの候補手を強制し、その後の勝敗を探索 AI 同士で比較する
export class CounterfactualPool extends WorkerPool<CounterfactualJob, CounterfactualResult> {
  constructor(size?: number) {
    super(new URL('./counterfactual-worker.ts', import.meta.url), size);
  }
}
