import { parentPort } from 'node:worker_threads';
import { type CounterfactualJob, evaluateCounterfactual } from '../../src/guide/counterfactual';

parentPort!.on('message', (job: CounterfactualJob) => {
  parentPort!.postMessage(evaluateCounterfactual(job));
});
