import { parentPort } from 'node:worker_threads';
import { type GuideJob, playGuide } from '../../src/guide/match';

parentPort!.on('message', (job: GuideJob) => {
  parentPort!.postMessage(playGuide(job));
});
