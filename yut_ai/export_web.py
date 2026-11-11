"""Export only inference weights, never optimizer/replay, for the static web app."""
import argparse,json,hashlib,io
from pathlib import Path
import torch
from .player import SCHEMA,STATE_DIM,ACTION_DIM

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,default=Path('web/model.json'));a=p.parse_args()
 raw=a.checkpoint.read_bytes()
 data=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
 if data['schema']!=SCHEMA or data['config']['state_dim']!=STATE_DIM or data['config']['action_dim']!=ACTION_DIM: raise ValueError('Incompatible game model')
 result={'schema':SCHEMA,'rules':'local-play-v1','training_search_version':data.get('search_version','legacy-or-unknown'),'config':data['config'],'step':data['step'],'games':data.get('run',{}).get('completed'),'sha256':hashlib.sha256(raw).hexdigest(),'weights':{k:v.tolist() for k,v in data['model'].items()}}
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,separators=(',',':')));tmp.replace(a.output)
 print(json.dumps({'output':str(a.output),'step':result['step'],'games':result['games']}))
if __name__=='__main__':main()
