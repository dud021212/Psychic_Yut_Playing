import fs from 'node:fs';import assert from 'node:assert/strict';import {Game,features} from '../web/engine.mjs';import {Network} from '../web/network.mjs';
let {rows,model}=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),net=new Network(model);
function close(a,b){if(Array.isArray(a)){assert.equal(a.length,b.length);a.forEach((x,i)=>close(x,b[i]))}else assert.ok(Math.abs(a-b)<2e-5,`${a} != ${b}`)}
for(let [i,r] of rows.entries()){try{let g=Game.from(r.before);assert.deepEqual(g.legal(),r.actions);close(features(g),r.features);let actual=net.evaluate(g,g.legal());close(actual.policy,r.network.policy);close(actual.value,r.network.value);g.apply(r.action,()=>.5,r.event);assert.deepEqual(JSON.parse(JSON.stringify(g)),r.after)}catch(e){console.error('Fixture',i,r.action);throw e}}
console.log(`PASS ${rows.length} Python/JS transitions and network predictions`);

// Exercise the worker entry point with private opponent information removed.
globalThis.self = {};
globalThis.fetch = async () => ({json: async () => model});
const messages = [];
globalThis.postMessage = message => messages.push(message);
await import('../web/ai-worker.mjs');
await self.onmessage({data: {type: 'init'}});
assert.equal(messages.at(-1).type, 'ready');
const game = new Game();
const legal = game.legal();
game.hands[1] = [null, null];
game.beliefs[1] = null;
await self.onmessage({data: {type: 'think', id: 1, game}});
const reply = messages.at(-1);
assert.equal(reply.type, 'action', JSON.stringify(reply));
assert(legal.some(action => JSON.stringify(action) === JSON.stringify(reply.action)));
console.log('PASS worker with hidden opponent powers');
