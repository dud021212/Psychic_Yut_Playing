export const SOURCES=['do','gae','geol','yut','mo','backdo'], DIST=[1,2,3,4,5,-1];
export const act=(kind,...args)=>({kind,args});
export const count=(xs,x)=>xs.filter(v=>v===x).length;
export function pick(xs,weights=null,rng=Math.random){let w=weights??xs.map(()=>1), r=rng()*w.reduce((a,b)=>a+b,0);for(let i=0;i<xs.length;i++){r-=w[i];if(r<0)return xs[i]}return xs.at(-1)}
export function sample(xs,n,rng=Math.random){let a=[...xs],r=[];while(n--)r.push(a.splice(Math.floor(rng()*a.length),1)[0]);return r}
export function belief(hand){return Array.from({length:64},(_,i)=>Math.floor(i/8)===hand[0]||i%8===hand[1]?0:1/49)}
export function reveal(bs,cat,id){let a=bs.map((p,i)=>(cat===0?Math.floor(i/8):i%8)===id?p:0),z=a.reduce((s,x)=>s+x,0);if(!z)throw Error('추정 정보 불일치');return a.map(x=>x/z)}
export function backward(c){return ({0:[19,28],15:[14,24],22:[21,26]})[c]??[({20:5,21:20,23:22,24:23,25:10,26:25,27:22,28:27})[c]??c-1]}
const routeCache=new Map();
export function paths(start,m){let key=[start,m.source,m.distance,m.backstep].join(':');if(routeCache.has(key))return routeCache.get(key);if(start===29||(start===-1&&m.distance<0))return [];
function walk(c,n,first,path){if(!n||c===29)return [path];let opts;if(m.distance>0){opts=[c===-1?1:c===0?29:c===22&&[10,25,26].includes(start)?27:first&&[5,10,22].includes(c)?({5:20,10:25,22:27})[c]:({20:21,21:22,22:23,23:24,24:15,25:26,26:22,27:28,28:0})[c]??(c+1)%20]}else opts=backward(c);return opts.flatMap(d=>m.backstep&&m.distance<0&&start!==0&&d===0?[[...path,d]]:walk(d,n-1,false,[...path,d]))}let result=walk(start,Math.abs(m.distance),true,[]);routeCache.set(key,result);return result}
export class Game{
constructor(rng=Math.random){let a=sample([...Array(8).keys()],2,rng),b=sample([...Array(8).keys()],2,rng);this.hands=[[a[0],b[0]],[a[1],b[1]]];this.positions=[Array(4).fill(-1),Array(4).fill(-1)];this.used=[[false,false],[false,false]];this.beliefs=this.hands.map(belief);this.player=0;this.phase='power';this.moves=[];this.stored=[[],[]];this.extra=false;this.active=null;this.takeout=false;this.installations={};this.turns=0;this.winner=null}
static from(data){return Object.assign(Object.create(Game.prototype),structuredClone(data))}
clone(){return Game.from(this)}
groups(p,board=false){let seen=new Set(),a=[];this.positions[p].forEach((x,i)=>{if(x===29||(board&&x===-1)||x>=0&&seen.has(x))return;seen.add(x);a.push(i)});return a}
members(p,i){let pos=this.positions[p][i];return pos===-1?[i]:this.positions[p].flatMap((x,j)=>x===pos?[j]:[])}
empty(c){return this.installations[c]===undefined&&!this.positions.flat().includes(c)}
legal(){if(this.winner!==null)return [];let p=this.player,q=1-p;
if(this.phase==='throw')return [act('throw')];
if(this.phase==='power'){let a=[act('skip')],power=this.hands[p][0];if(!this.used[p][0]&&(power!==2||this.groups(p,true).length))a.push(act('power',0));if(this.used[p][1])return a;power=this.hands[p][1];let own=this.groups(p,true),other=this.groups(q,true);
if(power===0){let h=this.positions[p].indexOf(-1);if(h>=0)own.forEach(i=>a.push(act('power',1,i,h)))}
else if(power===1)other.forEach(j=>a.push(act('power',1,j)));
else if(power===2)own.forEach(i=>other.forEach(j=>a.push(act('power',1,i,j))));
else if(power===3)own.forEach(i=>{let cells=new Set();[1,2,3].forEach(n=>paths(this.positions[p][i],{distance:n}).forEach(path=>{let c=path.at(-1);if(c!==29&&this.empty(c))cells.add(c)}));other.forEach(j=>[...cells].sort((a,b)=>a-b).forEach(c=>a.push(act('power',1,i,j,c))))});
else if(power===4)own.forEach(i=>other.forEach(j=>backward(this.positions[q][j]).forEach(c=>{if(!this.positions[q].includes(c))a.push(act('power',1,i,j,c))})));
else if(power===5){if(own.length||other.length)a.push(act('power',1))}
else for(let c=1;c<29;c++)if(this.empty(c))a.push(act('power',1,c));return a}
let a=this.extra?[act('throw')]:[],seen=new Set();if(!this.takeout)this.moves.forEach((m,k)=>{let id=JSON.stringify(m);if(seen.has(id))return;seen.add(id);this.groups(p).forEach(i=>paths(this.positions[p][i],m).forEach(path=>a.push(act('move',k,i,path))))});return a.length?a:[act('end')]}
relocate(p,ids,d){ids.forEach(i=>this.positions[p][i]=d)}
land(ids,d,source=null){let p=this.player,q=1-p;if(d>=0&&d<29){let targets=this.positions[q].flatMap((x,i)=>x===d?[i]:[]);if(targets.length){if(source===null)throw Error('능력 직접 잡기 불가');this.relocate(q,targets,-1);this.extra=this.extra||!['yut','mo'].includes(source)}}this.relocate(p,ids,d)}
apply(a,rng=Math.random,forced=null){let p=this.player,q=1-p,event={player:p,action:a.kind,args:a.args};if(a.kind==='skip')this.phase='throw';
else if(a.kind==='power'){let [cat,...xs]=a.args,power=this.hands[p][cat];this.used[p][cat]=true;this.beliefs[q]=reveal(this.beliefs[q],cat,power);event.power=[cat,power];if(cat===0){this.active=power;this.takeout=power===6}else{
if(power===0)this.positions[p][xs[1]]=this.positions[p][xs[0]];
else if(power===1)this.relocate(q,this.members(q,xs[0]),-1);
else if(power===2){let [i,j]=xs,own=this.members(p,i),other=this.members(q,j),x=this.positions[p][i],y=this.positions[q][j];this.relocate(p,own,y);this.relocate(q,other,x)}
else if(power===3)this.relocate(q,this.members(q,xs[1]),xs[2]);
else if(power===4){let ids=this.members(p,xs[0]),c=xs[2],installation=this.installations[c];delete this.installations[c];if(installation==='gate'){this.relocate(q,this.positions[q].flatMap((x,i)=>x===0?[i]:[]),-1);this.relocate(p,ids,0)}else this.land(ids,installation==='bomb'?-1:c)}
else if(power===5){let groups=[0,1].flatMap(owner=>this.groups(owner,true).map(i=>[owner,this.members(owner,i)]));let cells=forced?.shuffle??sample([...Array(29).keys()].filter(c=>this.installations[c]===undefined),groups.length,rng);groups.forEach(([o,ids],i)=>this.relocate(o,ids,cells[i]));event.shuffle=cells}
else this.installations[xs[0]]=power===6?'gate':'bomb';}this.phase='throw'}
else if(a.kind==='throw'){let first=this.phase==='throw',power=first?this.active:null;if(!first)this.extra=false;let special={0:{do_and_gae:1},1:{geol:1},2:{backdo:1},4:{do:.5,mo:.5},5:{yut:.5,mo:.5}},dist=special[power]??{nak:.015,do:.985*3/16,gae:.985*6/16,geol:.985*4/16,yut:.985/16,mo:.985/16,backdo:.985/16};let outcome=forced?.outcome??pick(Object.keys(dist),Object.values(dist),rng);event.outcome=outcome;if(first){this.moves.push(...this.stored[p]);this.stored[p]=[]}if(outcome==='do_and_gae')this.moves.push({source:'do',distance:1,backstep:false},{source:'gae',distance:2,backstep:false});else if(outcome!=='nak'){let m={source:outcome,distance:DIST[SOURCES.indexOf(outcome)]*(power===7?-1:1),backstep:power===7};this.moves.push(m);if(power===3)this.moves.push({...m});this.extra=this.extra||['yut','mo'].includes(outcome)}this.phase='move'}
else if(a.kind==='move'){let [k,i,path]=a.args,m=this.moves.splice(k,1)[0],ids=this.members(p,i),dest=path.at(-1);for(let c of path){let ins=this.installations[c];if(ins){delete this.installations[c];dest=ins==='gate'?0:-1;break}}this.land(ids,dest,m.source)}
else if(a.kind==='end'){if(this.takeout)this.stored[p].push(...this.moves);this.moves=[];this.extra=false;this.active=null;this.takeout=false;this.player=q;this.phase='power';this.turns++}else throw Error('알 수 없는 행동');for(let owner of [0,1])if(this.positions[owner].every(x=>x===29))this.winner=owner;return event}
}
export function destination(g,a){let dest=a.args[2].at(-1);for(let c of a.args[2])if(g.installations[c]){dest=g.installations[c]==='gate'?0:-1;break}return dest}
export function features(g){let p=g.player,q=1-p,f=[];for(let o of [p,q])for(let c=-1;c<=29;c++)f.push(count(g.positions[o],c)/4);for(let c=0;c<29;c++)for(let k of ['gate','bomb'])f.push(+(g.installations[c]===k));for(let o of [p,q])f.push(...g.used[o].map(Number));f.push(...['power','throw','move'].map(x=>+(g.phase===x)),+g.extra,+g.takeout);for(let moves of [g.moves,g.stored[p],g.stored[q]])for(let s of SOURCES)for(let dir of [-1,1])f.push(Math.log1p(moves.filter(m=>m.source===s&&m.distance*dir>0).length));for(let i=0;i<8;i++)f.push(+(g.active===i));for(let power of g.hands[p])for(let i=0;i<8;i++)f.push(+(i===power));f.push(...g.beliefs[p]);return f}
export function actionFeatures(g,a){let f=Array(48).fill(0);f[['skip','power','throw','move','end'].indexOf(a.kind)]=1;let p=g.player;if(a.kind==='move'){let [k,i,path]=a.args,m=g.moves[k],d=destination(g,a);f[5+SOURCES.indexOf(m.source)]=1;f[11]=m.distance/5;f[12]=g.members(p,i).length/4;f[13]=(g.positions[p][i]+1)/30;f[14]=(d+1)/30;f[15]=d>=0&&d<29?count(g.positions[1-p],d)/4:0;f[16]=d>=0&&d<29?count(g.positions[p],d)/4:0;path.slice(0,5).forEach((c,j)=>f[17+j]=(c+1)/30)}else if(a.kind==='power'){let cat=a.args[0];f[22+cat*8+g.hands[p][cat]]=1;a.args.slice(1).forEach((x,j)=>f[38+j]=(x+1)/30)}return f}
