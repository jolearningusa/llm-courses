import copy,json,unittest
from pathlib import Path
from demo_replay import parse_stdout,collect

TEXT='after SFT on 30 prompts: mean reward 0.273 (length: correct 3.93, incorrect 4.80)\n     GRPO after 300 steps: mean reward 0.506 (length: correct 2.43, incorrect 2.79)\n Dr. GRPO after 300 steps: mean reward 0.489 (length: correct 2.46, incorrect 2.49)\n'

class DemoReplayControls(unittest.TestCase):
    def test_printed_precision_and_stages(self):
        self.assertEqual(parse_stdout(TEXT)['SFT'],[0.273,3.93,4.8])
        for bad in [TEXT.splitlines()[0],TEXT.replace('0.506','1.500'),TEXT.replace('2.43','9.00'),TEXT.replace('GRPO after 300','OTHER after 300')]:
            with self.assertRaises(ValueError):parse_stdout(bad)

    def test_worker_completeness_and_uniqueness(self):
        protocol=json.loads(Path(__file__).with_name('demo_replay_protocol.json').read_text())
        rows=[{'lane':l,'seed':s,'stdout':TEXT,'printed_metrics':parse_stdout(TEXT)} for l in protocol['lanes'] for s in protocol['seeds']]
        self.assertEqual(len(collect(rows,protocol)),2)
        for bad in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):collect(bad,protocol)

    def test_parsed_metric_mutation_rejected(self):
        protocol={'lanes':{'x':{}},'seeds':[0]};row={'lane':'x','seed':0,'stdout':TEXT,'printed_metrics':parse_stdout(TEXT)}
        row['printed_metrics']['GRPO'][0]=0.9
        with self.assertRaises(ValueError):collect([row],protocol)

if __name__=='__main__':unittest.main()
