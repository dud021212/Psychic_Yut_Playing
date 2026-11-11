import {Game,belief,reveal,pick} from './engine.mjs';
import {Network} from './network.mjs';
let model;
function rollout(g){
  const actions=g.legal();
  if(actions.length===1)return actions[0];
  return pick(actions,model.evaluate(g,actions).policy);
}
function determinize(g,root){let t=g.clone(),index=pick([...Array(64).keys()],g.beliefs[root]);t.hands[1-root]=[Math.floor(index/8),index%8];t.beliefs[1-root]=belief(t.hands[1-root]);for(let cat=0;cat<2;cat++)if(g.used[root][cat])t.beliefs[1-root]=reveal(t.beliefs[1-root],cat,g.hands[root][cat]);return t}
self.onmessage=async ({data})=>{try{if(data.type==='init'){model=new Network(await (await fetch('./model.json')).json());postMessage({type:'ready',games:model.data.games,step:model.data.step});return}let g=Game.from(data.game),root=g.player,actions=g.legal();if(!model)throw Error('모델을 아직 불러오지 못했습니다');if(actions.length===1){postMessage({type:'action',id:data.id,action:actions[0]});return}for(let a of actions)if(a.kind==='move'){let t=determinize(g,root);t.apply(a);if(t.winner===root){postMessage({type:'action',id:data.id,action:a});return}}
let {policy}=model.evaluate(g,actions),visits=actions.map(()=>0),totals=actions.map(()=>0),budget=Math.max(256,4*actions.length);
for(let it=0;it<budget;it++){let index=it<actions.length?it:totals.map((v,i)=>v/visits[i]+1.5*policy[i]*Math.sqrt(it)/(1+visits[i])).reduce((best,v,i,a)=>v>a[best]?i:best,0);let t=determinize(g,root);t.apply(actions[index]);let n=0;while(t.winner===null&&n++<256)t.apply(rollout(t));let value=t.winner!==null?(t.winner===root?1:-1):model.evaluate(t,t.legal()).value*(t.player===root?1:-1);visits[index]++;totals[index]+=value;if(it%16===0)postMessage({type:'progress',id:data.id,done:it+1,total:budget})}let best=visits.reduce((b,v,i)=>v>visits[b]?i:b,0);postMessage({type:'action',id:data.id,action:actions[best]})}catch(e){postMessage({type:'error',message:e.message,id:data.id})}};
