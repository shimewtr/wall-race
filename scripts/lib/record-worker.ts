import { parentPort } from 'node:worker_threads';
import { type RecordJob, playRecorded } from '../../src/train/record';

parentPort!.on('message', (job: RecordJob) => {
  parentPort!.postMessage(playRecorded(job));
});
