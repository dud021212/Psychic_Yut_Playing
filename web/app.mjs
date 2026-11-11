import {Game,act,count,destination} from './engine.mjs';
const $=id=>document.getElementById(id),names={do:'도',gae:'개',geol:'걸',yut:'윷',mo:'모',backdo:'뒷도',nak:'낙',do_and_gae:'도 + 개'};
const powers=[['개와 돼지의 시간','100마력','후진 기어','디스 이즈 더블더블','모 아니면 도','마이더스의 손','테이크 아웃','백스텝'],['업고 가실게요','강제 귀가','자리 좀 바꿔주세요','제 앞에 놔주세요','백허그','밥상 뒤집기','결승점이 코앞이네','폭발물 주의']];
const desc=[['도와 개를 함께 얻습니다. 낙 없음.','걸 확정. 낙 없음.','뒷도 확정. 판 위 말이 필요합니다. 낙 없음.','최초 투척 결과를 2개 얻습니다. 낙 가능.','모·도 각각 50%. 낙 없음.','윷·모 각각 50%. 낙 없음.','이번 차례에는 이동하지 않고 모든 결과를 다음 차례로 저장합니다.','최초 결과만큼 후진합니다. 뒷도는 1칸 전진. 낙 가능.'],['판 위 자기 묶음에 출발 전 말 하나를 업습니다.','상대 묶음을 출발 전으로 보냅니다.','자기 묶음과 상대 묶음의 위치를 교환합니다.','상대 묶음을 자기 말 앞 3칸 이내 빈 칸으로 옮깁니다.','자기 묶음을 상대 바로 뒤로 옮깁니다.','모든 판 위 묶음을 서로 다른 칸에 재배치합니다.','빈 칸에 게이트를 설치합니다. 양쪽 모두 사용 가능.','빈 칸에 폭탄을 설치합니다. 양쪽 모두 영향.']];
let g,human=0,log=[],worker,ready=false,busy=false,token=0,selection=null,error='',presenting=false;
const posName=c=>c===-1?'출발 대기':c===29?'완주':c===0?'결승':c===22?'중앙':`${c}번 칸`;
const moveName=m=>`${names[m.source]} · ${m.distance<0?'후진':'전진'} ${Math.abs(m.distance)}칸`;
const pieceName=(owner,i)=>`${owner===human?'내':'AI'} 말 ${String.fromCharCode(65+i)}${g.members(owner,i).length>1?' 묶음 ×'+g.members(owner,i).length:''} (${posName(g.positions[owner][i])})`;
function button(text,fn,disabled=false,cls=''){let b=document.createElement('button');b.textContent=text;b.onclick=fn;b.disabled=disabled;b.className=cls;return b}
function save(){try{localStorage.setItem('psychic-yut-v1',JSON.stringify({g,human,log}))}catch{}}
function say(s){log.push(s);if(log.length>100)log.shift()}
function start(){if(g&&g.winner===null&&!confirm('진행 중인 대국을 끝내고 새 대국을 시작할까요?'))return;token++;worker?.terminate();ready=false;busy=false;presenting=false;error='';$('action-notice').textContent='';g=new Game();human=crypto.getRandomValues(new Uint32Array(1))[0]%2;selection=null;log=[];say(`새 대국 · ${human===0?'내가 선공':'AI가 선공'} · 능력 새로 배정`);save();boot();render()}
function boot(){worker=new Worker('./ai-worker.mjs',{type:'module'});worker.onmessage=({data})=>{if(data.type==='ready'){ready=true;$('model-info').textContent=`탑재 모델: 누적 ${data.games??'?'}판 · 학습 ${data.step}회. 모델 갱신은 재내보내기 후 반영됩니다.`;render();advance();return}if(data.type==='error'&&data.id===undefined){error='모델 로딩 실패: '+data.message;busy=false;render();return}if(data.id!==token)return;if(data.type==='progress'){$('progress').max=data.total;$('progress').value=data.done;return}if(data.type==='error'){error=`AI 오류: ${data.message}`;busy=false;render();return}if(data.type==='action'){busy=false;commit(data.action)}};worker.onerror=e=>{error='AI 파일 실행 실패: '+e.message;busy=false;render()};worker.postMessage({type:'init'})}
async function commit(a){if(presenting||busy||g.winner!==null)return;if(!g.legal().some(x=>JSON.stringify(x)===JSON.stringify(a))){error='선택 가능한 행동이 바뀌었습니다.';selection=null;render();return}let who=g.player===human?'나':'AI',before=g.clone(),event=g.apply(a);if(a.kind==='throw')say(`${who}: ${names[event.outcome]}${event.outcome==='nak'?' — 이번 투척 무효':''}`);if(a.kind==='move')say(`${who}: ${moveName(before.moves[a.args[0]])} → ${posName(g.positions[before.player][a.args[1]])}`);if(a.kind==='power')say(`${who}: ${powers[event.power[0]][event.power[1]]} 사용`);if(a.kind==='end')say(`${who} 차례 종료`);if(g.winner!==null)say(g.winner===human?'승리! 내 말 4개 완주':'AI 승리 · AI 말 4개 완주');selection=null;save();const generation=token;presenting=true;try{await presentAction(before,a,event,generation)}finally{if(generation===token){presenting=false;render();advance()}}}
function advance(){if(!ready||busy||presenting||error||g.winner!==null||g.player===human)return;busy=true;render();let snapshot=structuredClone(g);/* Never send the human's unused private powers/posterior to AI. */snapshot.hands[human]=snapshot.hands[human].map((v,c)=>g.used[human][c]?v:null);snapshot.beliefs[human]=null;worker.postMessage({type:'think',id:++token,game:snapshot})}
let targetClicks=new Map(),previewRoute=[];
const choiceArgs = a => a.kind === 'move' ? [a.args[1], a.args[0], a.args[2]] : a.args;
function showChoices(){let area=$('choices'),controls=$('controls');area.replaceChildren();area.className='';controls.replaceChildren();$('preview').textContent='';targetClicks=new Map();previewRoute=[];let acts=g.legal();if(g.winner!==null){$('instruction').textContent=g.winner===human?'축하합니다. 모든 말을 완주시켰습니다.':'다음 대국에서 다시 도전해 보세요.';area.append(button('새 대국',start,false,'primary'));return}if(presenting){$('instruction').textContent='행동을 보여주는 중입니다.';return}if(g.player!==human||!ready||error){$('instruction').textContent=error||'AI가 수를 선택하고 있습니다. 잠시 기다려 주세요.';if(error)area.append(button('AI 다시 연결',()=>{worker?.terminate();error='';busy=false;ready=false;boot();render()}));return}
if(!selection){$('instruction').textContent=g.phase==='power'?'초능력을 사용할지 선택하세요. 사용하지 않고 던질 수도 있습니다.':g.phase==='throw'?'윷을 던져 이동 결과를 받으세요.':g.takeout?'테이크 아웃: 추가 투척을 처리한 뒤 결과를 저장합니다.':'이동할 말을 먼저 선택하세요.';
if(g.phase==='power'){for(let c=0;c<2;c++)area.append(button(powers[c][g.hands[human][c]],()=>{selection={kind:'power',prefix:[c]};render()},!acts.some(a=>a.kind==='power'&&a.args[0]===c)));area.append(button('능력 없이 던지기',async()=>{const generation=token;await commit(act('skip'));if(generation===token)await commit(act('throw'))},false,'primary'))}
else {if(acts.some(a=>a.kind==='throw'))area.append(button(g.phase==='throw'?'윷 던지기':'추가 투척 사용',()=>commit(act('throw')),false,'primary'));const moves=acts.filter(a=>a.kind==='move');if(moves.length){let grid=document.createElement('div');grid.className='piece-grid';let seen=new Set();moves.forEach(a=>{let i=a.args[1];if(seen.has(i))return;seen.add(i);let choose=()=>{selection={kind:'move',prefix:[i]};render()};grid.append(button(pieceName(human,i),choose));let cell=g.positions[human][i];if(cell>=0)targetClicks.set(cell,choose)});area.append(grid)}if(acts.some(a=>a.kind==='end'))area.append(button(g.takeout?'결과 저장하고 차례 종료':'차례 종료',()=>commit(act('end')),false,'primary'))}return}
let filtered=acts.filter(a=>a.kind===selection.kind&&selection.prefix.every((v,i)=>JSON.stringify(choiceArgs(a)[i])===JSON.stringify(v))),n=selection.prefix.length;if(!filtered.length){selection=null;showChoices();return}
let complete=filtered.every(a=>a.args.length===n);if(complete){let a=filtered[0];$('instruction').textContent='선택한 행동을 확인하세요.';if(a.kind==='move'){previewRoute=a.args[2];$('preview').textContent=`${moveName(g.moves[a.args[0]])} · ${pieceName(human,a.args[1])} → ${posName(destination(g,a))}`}else {let power=g.hands[human][a.args[0]],xs=a.args.slice(1),detail='';if(a.args[0]===1){if(power===0)detail=pieceName(human,xs[0])+' ← '+pieceName(human,xs[1]);else if(power===1)detail=pieceName(1-human,xs[0]);else if(power>=2&&power<=4)detail=pieceName(human,xs[0])+' / '+pieceName(1-human,xs[1])+(xs.length>2?' → '+posName(xs[2]):'');else if(power>=6)detail=posName(xs[0]);} $('preview').textContent=powers[a.args[0]][power]+' 사용'+(detail?' · '+detail:'');}area.append(button('확인 · 실행',()=>commit(a),false,'primary'))}
else {let role;if(selection.kind==='move')role=n===1?'result':'path';else {let pow=g.hands[human][selection.prefix[0]];role=({0:['own','home'],1:['enemy'],2:['own','enemy'],3:['own','enemy','cell'],4:['own','enemy','cell'],5:[],6:['cell'],7:['cell']})[pow]?.[n-1]??'cell'}$('instruction').textContent=({result:'선택한 말에 사용할 이동 결과를 고르세요.',own:'이동하거나 사용할 내 말을 선택하세요.',enemy:'대상 AI 묶음을 선택하세요.',home:'업을 출발 전 말을 선택하세요.',path:'이동 경로를 선택하세요.',cell:'목적지 또는 설치할 칸을 선택하세요.'})[role];if(['own','home','enemy'].includes(role))area.className='piece-grid';if(selection.kind==='move')$('preview').textContent=pieceName(human,selection.prefix[0]);let used=new Set();for(let a of filtered){let val=choiceArgs(a)[n],key=JSON.stringify(val);if(used.has(key))continue;used.add(key);let label,cell=null;if(['own','home','enemy'].includes(role)){let o=role==='enemy'?1-human:human;label=pieceName(o,val);cell=g.positions[o][val]}else if(role==='result'){label=moveName(g.moves[val])}else if(role==='path'){label=`${val.map(posName).join(' → ')}${destination(g,a)!==val.at(-1)?' · 설치물로 '+posName(destination(g,a)):''}`;cell=destination(g,a)}else {label=posName(val);cell=val}let choose=()=>{selection.prefix.push(val);render()};area.append(button(label,choose));if(cell!==null&&cell>=0&&cell<29&&!targetClicks.has(cell))targetClicks.set(cell,choose)}}
controls.append(button('이전',()=>{selection.prefix.pop();if(!selection.prefix.length)selection=null;render()}),button('선택 취소',()=>{selection=null;render()}))}
const coords=[[540,540]];for(let i=1;i<=4;i++)coords[i]=[540,540-96*i];coords[5]=[540,60];for(let i=6;i<=9;i++)coords[i]=[540-96*(i-5),60];coords[10]=[60,60];for(let i=11;i<=14;i++)coords[i]=[60,60+96*(i-10)];coords[15]=[60,540];for(let i=16;i<=19;i++)coords[i]=[60+96*(i-15),540];Object.assign(coords,{20:[460,140],21:[380,220],22:[300,300],23:[220,380],24:[140,460],25:[140,140],26:[220,220],27:[380,380],28:[460,460]});

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function presentAction(before, action, event, generation) {
  const who = before.player === human ? '나' : 'AI';
  let text = log.at(-1) ?? '';
  if (action.kind === 'skip') text = `${who}: 초능력 없이 윷을 던집니다`;
  const notice = $('action-notice');
  notice.textContent = text;
  notice.className = before.player === human ? 'action-notice blue' : 'action-notice orange';
  notice.animate([{opacity: .3, transform: 'translateY(6px)'}, {opacity: 1, transform: 'translateY(0)'}], {duration: 220});
  $('turn').textContent = `${who}의 행동을 보여주는 중`;
  $('progress').hidden = true;
  $('choices').replaceChildren();
  $('controls').replaceChildren();
  $('instruction').textContent = text;
  targetClicks.clear();
  if (action.kind === 'move') {
    const [move, piece, path] = action.args;
    const route = [before.positions[before.player][piece]];
    for (const cell of path) {
      route.push(cell);
      if (before.installations[cell]) {
        route.push(before.installations[cell] === 'gate' ? 0 : -1);
        break;
      }
    }
    const marker = document.createElement('div');
    const own = before.player === human;
    marker.className = 'moving-piece ' + (own ? 'human' : 'enemy');
    const size = before.members(before.player, piece).length;
    marker.textContent = (own ? '●' : '◆') + (size > 1 ? '×' + size : '');
    $('board').append(marker);
    const origin = route[0];
    const oldCell = origin >= 0 && origin < 29 ? $('cells').children[origin] : null;
    if (oldCell) oldCell.style.opacity = '.25';
    const point = cell => cell === -1 ? [570, 585] : cell === 29 ? [570, 550] : coords[cell];
    const frames = route.map(cell => {const [x,y] = point(cell);return {left: x/6+'%', top: y/6+'%'}});
    try {
      const animation = marker.animate(frames, {duration: matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : Math.max(500, (route.length-1)*230), easing: 'linear', fill: 'forwards'});
      await animation.finished;
    } finally {marker.remove();if(oldCell)oldCell.style.opacity=''}
  }
  if (generation !== token) return;
  render();
  let powerEffect;
  if (action.kind === 'power') {
    powerEffect=document.createElement('div');
    powerEffect.className='power-effect '+(before.player===human?'own-effect':'enemy-effect');
    const label=document.createElement('small');label.textContent=who+' · 초능력 발동';
    const title=document.createElement('strong');title.textContent=powers[event.power[0]][event.power[1]];
    const detail=document.createElement('p');detail.textContent=desc[event.power[0]][event.power[1]];
    powerEffect.append(label,title,detail);$('board').append(powerEffect);
    for (const owner of [0,1]) g.positions[owner].forEach((cell,i) => {
      if (cell >= 0 && cell < 29 && cell !== before.positions[owner][i])
        $('cells').children[cell].animate([{outline:'5px solid #e6b000'}, {outline:'5px solid transparent'}], {duration:1000});
    });
  }
  // Keep each outcome visible before requesting the AI's next action.
  try {await pause(action.kind === 'power' ? 1800 : action.kind === 'throw' ? 1100 : action.kind === 'skip' ? 250 : 450)}
  finally {powerEffect?.remove()}
}

function render(){if(!g)return;let yours=g.player===human;$('status').className=g.winner!==null?'':yours?'yours':'ai';$('turn').textContent=g.winner!==null?(g.winner===human?'승리했습니다!':'AI가 승리했습니다'):error?'대국을 진행할 수 없습니다':!ready?'AI 모델을 불러오는 중':presenting?'행동 진행 중':yours?'지금은 내 차례입니다':'AI 차례 · 수를 읽는 중';$('hint').textContent=error||`최고 난이도 · ${g.turns+1}번째 차례 · ${yours?'시간 제한 없이 천천히 선택하세요.':'내 차례가 되면 선택 버튼이 활성화됩니다.'}`;$('progress').hidden=!busy;$('human-score').textContent=`● 나 · 완주 ${count(g.positions[human],29)}/4`;$('ai-score').textContent=`◆ AI · 완주 ${count(g.positions[1-human],29)}/4`;showChoices();let cells=$('cells');cells.replaceChildren();coords.forEach(([x,y],c)=>{let own=count(g.positions[human],c),enemy=count(g.positions[1-human],c),ins=g.installations[c];let b=button('',()=>targetClicks.get(c)?.());b.className='cell'+(own?' human':enemy?' enemy':ins?' install':'')+(targetClicks.has(c)?' target':'')+(previewRoute.includes(c)?' route':'');b.style.left=x/6+'%';b.style.top=y/6+'%';b.textContent=own?'●'+(own>1?'×'+own:''):enemy?'◆'+(enemy>1?'×'+enemy:''):ins==='gate'?'G':ins==='bomb'?'B':c===0?'결승':c===22?'중앙':String(c);b.title=`${posName(c)}${own?' 내 말 '+own+'개':enemy?' AI 말 '+enemy+'개':ins==='gate'?' 게이트':ins==='bomb'?' 폭탄':''}`;b.setAttribute('aria-label',b.title);b.tabIndex=targetClicks.has(c)?0:-1;cells.append(b)});$('home').replaceChildren();for(let owner of [human,1-human]){let span=document.createElement('span');span.textContent=`${owner===human?'내':'AI'} 출발 대기: ${count(g.positions[owner],-1)}개`;span.className=owner===human?'blue':'orange';$('home').append(span)}
$('powers').replaceChildren();for(let c=0;c<2;c++){let div=document.createElement('div');div.className='power';let strong=document.createElement('strong');strong.textContent=`${powers[c][g.hands[human][c]]} · ${g.used[human][c]?'사용 완료':'사용 가능'}`;let p=document.createElement('p');p.textContent=desc[c][g.hands[human][c]];div.append(strong,p);$('powers').append(div)}$('enemy-powers').replaceChildren();for(let c=0;c<2;c++){const revealed=g.used[1-human][c],card=document.createElement('div');card.className='enemy-power-card'+(revealed?' revealed':'');const category=document.createElement('small');category.textContent=c===0?'투척 초능력':'말·설치 초능력';const title=document.createElement('strong');title.textContent=revealed?powers[c][g.hands[1-human][c]]:'미공개';const detail=document.createElement('p');detail.textContent=revealed?'사용 완료 · '+desc[c][g.hands[1-human][c]]:'사용하면 이 자리에 공개됩니다.';card.append(category,title,detail);$('enemy-powers').append(card)}$('results').replaceChildren();if(!g.moves.length)$('results').textContent='없음';else g.moves.forEach(m=>{let s=document.createElement('span');s.className='chip';s.textContent=moveName(m);$('results').append(s)});$('saved').textContent='내 저장 결과: '+(g.stored[human].map(moveName).join(', ')||'없음');$('log').replaceChildren(...log.slice().reverse().map(s=>{let li=document.createElement('li');li.textContent=s;return li}))}
$('new').onclick=start;
try{let saved=JSON.parse(localStorage.getItem('psychic-yut-v1'));if(saved?.g&&[0,1].includes(saved.human)){g=Game.from(saved.g);human=saved.human;log=saved.log??[]}}catch{}
if(!g){g=new Game();human=crypto.getRandomValues(new Uint32Array(1))[0]%2;say(`새 대국 · ${human===0?'내가 선공':'AI가 선공'} · 능력 새로 배정`);save()}render();boot();
