import json
from pathlib import Path
import tempfile
import unittest
from yut_ai.evaluate import previous_pair

class PreviousTests(unittest.TestCase):
    def test_select_latest_and_reject_unfinished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.pt'
            with self.assertRaises(ValueError): previous_pair(model)
            for name in ('20260101-old', '20260102-new'):
                folder = root / 'model.history' / name
                folder.mkdir(parents=True)
                (folder/'session.json').write_text(json.dumps({'status':'complete'}))
                (folder/'before.pt').touch(); (folder/'after.pt').touch()
            after, before = previous_pair(model)
            self.assertEqual(Path(after).parent.name, '20260102-new')
            self.assertEqual(Path(before).name, 'before.pt')
            (folder/'session.json').write_text(json.dumps({'status':'training_interrupted'}))
            with self.assertRaises(ValueError): previous_pair(model)
            (folder/'session.json').write_text(json.dumps({'status':'evaluation_interrupted'}))
            self.assertEqual(previous_pair(model), (after, before))
            (folder/'after.pt').unlink()
            with self.assertRaises(ValueError): previous_pair(model)

if __name__ == '__main__': unittest.main()
