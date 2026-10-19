import { parentPort } from 'node:worker_threads';
import { type DiscoveryJob, discoverGame } from '../../src/guide/discover';

parentPort!.on('message', (job: DiscoveryJob) => {
  parentPort!.postMessage(discoverGame(job));
});
