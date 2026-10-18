// 対局をワーカースレッドに分散して全コアで回す。

import { availableParallelism } from 'node:os';
import { Worker } from 'node:worker_threads';
import type { Weights } from '../../src/engine/ai';
import type { MatchConfig } from '../../src/train/match';

export interface PairJob {
  a: Partial<Weights>;
  b: Partial<Weights>;
  opts: Omit<MatchConfig, 'weights'>;
}

export class PairPool {
  private workers: Worker[] = [];
  private idle: Worker[] = [];
  private queue: Array<{ job: PairJob; resolve: (v: number) => void; reject: (e: unknown) => void }> = [];

  constructor(size = availableParallelism()) {
    for (let i = 0; i < size; i++) {
      // tsx の環境を引き継いで TypeScript のままワーカーを起動する
      const w = new Worker(new URL('./pool-worker.ts', import.meta.url), { execArgv: ['--import', 'tsx'] });
      this.workers.push(w);
      this.idle.push(w);
    }
  }

  get size(): number {
    return this.workers.length;
  }

  run(job: PairJob): Promise<number> {
    return new Promise((resolve, reject) => {
      this.queue.push({ job, resolve, reject });
      this.pump();
    });
  }

  runAll(jobs: PairJob[]): Promise<number[]> {
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
      const onMessage = (v: number) => {
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
