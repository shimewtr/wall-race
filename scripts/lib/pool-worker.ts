import { parentPort } from 'node:worker_threads';
import { playPair } from '../../src/train/match';
import type { PairJob } from './pool';

parentPort!.on('message', (job: PairJob) => {
  parentPort!.postMessage(playPair(job.a, job.b, job.opts));
});
