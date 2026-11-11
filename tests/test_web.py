import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
import torch
from yut_ai.game import Game
from yut_ai.belief import PowerBelief
from yut_ai.player import features, action_features, SCHEMA
from yut_ai.model import ModelConfig, PolicyValueNet

ROOT=Path(__file__).resolve().parents[1]
def snapshot(g):
 d=dict(g.__dict__)
 d['beliefs']=[list(b.features()) for b in g.beliefs]
 d['moves']=[m.__dict__ for m in g.moves]
 d['stored']=[[m.__dict__ for m in ms] for ms in g.stored]
 return json.loads(json.dumps(d))

@unittest.skipUnless(shutil.which('node'),'Node needed for JS parity')
class WebTests(unittest.TestCase):
 def test_python_js_engine_and_network_parity(self):
  rows=[]; rng=random.Random(909)
  torch.manual_seed(909)
  model=PolicyValueNet(ModelConfig(253,48)).eval()
  def record(g,a):
   before=snapshot(g); legal=g.legal_actions(); before_features=features(g)
   with torch.inference_mode():
    p,v=model.predict(torch.tensor([features(g)]),torch.tensor([[action_features(g,x) for x in legal]]),torch.ones(1,len(legal),dtype=torch.bool))
   expected={'policy':p[0].tolist(),'value':float(v[0])}
   event=g.apply(a,rng)
   rows.append({'before':before,'actions':[{'kind':x.kind,'args':x.args} for x in legal], 'action':{'kind':a.kind,'args':a.args},'event':event,'after':snapshot(g),'features':before_features, 'network':expected})
  from yut_ai.rules import MoveResult
  for start in (5,10,20,21,22,25,26):
   for source,distance in (('yut',4),('mo',5)):
    g=Game.new(rng);g.phase='move';g.positions[0][0]=start
    g.moves=[MoveResult(source,distance)]
    for a in g.legal_actions():record(g.clone(),a)
  for category in (0,1):
   for power in range(8):
    g=Game.new(rng);g.hands[0][category]=power;g.hands[1][category]=(power+1)%8;g.beliefs=[PowerBelief(*h) for h in g.hands];g.positions=[[1,1,-1,-1],[3,3,10,-1]]
    for a in g.legal_actions():
     if a.kind=='power' and a.args[0]==category:record(g.clone(),a)
  for seed in range(3):
   g=Game.new(random.Random(seed))
   from yut_ai.player import rollout_action
   for _ in range(600):
    record(g,rollout_action(g,rng))
    if g.winner is not None:break
  with tempfile.TemporaryDirectory() as temp:
   file=Path(temp)/'cases.json';file.write_text(json.dumps({'rows':rows,'model':{'schema':SCHEMA,'config':{'state_dim':253},'weights':{k:v.tolist() for k,v in model.state_dict().items()}}}))
   result=subprocess.run(['node',str(ROOT/'tests/web_parity.mjs'),str(file)],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)

