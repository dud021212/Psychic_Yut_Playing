import unittest
from unittest.mock import patch
import numpy as np
import torch
from yut_ai.model import ModelConfig, PolicyValueNet
from yut_ai.coreml_backend import ExportNetwork, CoreMLPredictor
from yut_ai.player import evaluate, features, action_features
from yut_ai.game import Game
import random

class CoreMLAdapterTests(unittest.TestCase):
    def test_unmasked_export_matches_torch_and_chunked_global_softmax(self):
        torch.manual_seed(11)
        model=PolicyValueNet(ModelConfig(253,48)).eval()
        wrapper=ExportNetwork(model).eval()
        state=torch.randn(1,253)
        for n in (1,31,32,33,97):
            actions=torch.randn(1,n,48)
            with torch.inference_mode():
                logits,value=wrapper(state,actions)
                p,v=model.predict(state,actions,torch.ones(1,n,dtype=torch.bool))
            torch.testing.assert_close(logits.softmax(-1),p)
            torch.testing.assert_close(value,v)
            class Runtime:
                def predict(self,data):
                    with torch.inference_mode():
                        a,b=wrapper(torch.from_numpy(data['states']),torch.from_numpy(data['actions']))
                    return {'logits':a.numpy(),'value':b.numpy()}
            predictor=CoreMLPredictor.__new__(CoreMLPredictor)
            predictor.model=Runtime();predictor.chunk=32;predictor.state_dim=253;predictor.action_dim=48
            actual,estimate=predictor.evaluate_features(state[0].numpy(),actions[0].numpy())
            np.testing.assert_allclose(actual,p[0].numpy(),atol=1e-6)
            self.assertAlmostEqual(estimate,float(v[0]),places=6)

    def test_search_adapter_dispatch_without_torch_parameters(self):
        game=Game.new(random.Random(1));actions=game.legal_actions()
        class Adapter:
            def evaluate_features(self,s,a):
                self.inputs=(s,a)
                return [1/len(a)]*len(a),0.25
        adapter=Adapter()
        policy,value=evaluate(adapter,game,actions)
        self.assertEqual(adapter.inputs,(features(game),[action_features(game,a) for a in actions]))
        self.assertEqual(value,0.25)
        self.assertAlmostEqual(sum(policy),1)

    def test_training_rounds_reconvert_updated_weights(self):
        from yut_ai.parallel import GameWorkers
        model=PolicyValueNet(ModelConfig(253,48))
        snapshots=[]
        def export(source,path):
            snapshots.append(next(source.parameters()).detach().clone())
            return str(path)
        with patch('yut_ai.coreml_backend.export_model',side_effect=export) as convert, patch('yut_ai.coreml_backend.CoreMLPredictor') as load, patch('yut_ai.selfplay.play',return_value=('game',[],[])) as play:
            workers=GameWorkers(1,'coreml-ne')
            args=[(1,4,None,100,8),(2,4,None,100,8)]
            workers.generate(model,args)
            with torch.no_grad():next(model.parameters()).add_(1)
            workers.generate(model,args)
            self.assertEqual(convert.call_count,2)
            self.assertEqual(load.call_count,2)
            self.assertEqual(play.call_count,4)
            self.assertFalse(torch.equal(*snapshots))
            self.assertNotEqual(load.call_args_list[0].args,load.call_args_list[1].args)

    def test_evaluation_worker_uses_coreml_models(self):
        import yut_ai.evaluate as module
        try:
            with patch.object(module,'initialize_worker'), patch('yut_ai.coreml_backend.CoreMLPredictor') as load:
                module.initialize_evaluation(['one.mlpackage',None],'coreml-ne')
                load.assert_called_once_with('one.mlpackage')
                self.assertIs(module._evaluation_models[0],load.return_value)
                self.assertIsNone(module._evaluation_models[1])
        finally:
            module._evaluation_models=None
