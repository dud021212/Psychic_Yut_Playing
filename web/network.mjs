import {features,actionFeatures} from './engine.mjs';
export class Network{
constructor(data){if(data.schema!=='local-play-v1-public-belief-features-v1'||data.config.state_dim!==253)throw Error('모델 형식이 맞지 않습니다');this.data=data;this.w=data.weights}
linear(x,key,relu=false){let ws=this.w[key+'.weight'],bs=this.w[key+'.bias'];return ws.map((row,i)=>{let v=bs?.[i]??0;for(let j=0;j<row.length;j++)v+=row[j]*x[j];return relu?Math.max(0,v):v})}
evaluate(g,actions){let h=this.linear(this.linear(features(g),'encoder.0',true),'encoder.2',true),v=Math.tanh(this.linear(this.linear(h,'value.0',true),'value.2')[0]),s=this.linear(h,'state_policy');let logits=actions.map(a=>{let u=this.linear(actionFeatures(g,a),'action_policy');return this.linear(s.map((x,i)=>Math.max(0,x+u[i])),'score')[0]});let top=Math.max(...logits),exp=logits.map(x=>Math.exp(x-top)),sum=exp.reduce((a,b)=>a+b,0);return {policy:exp.map(x=>x/sum),value:v}}
}
