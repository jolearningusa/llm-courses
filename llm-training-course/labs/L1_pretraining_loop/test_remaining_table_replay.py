import copy,json,unittest
from pathlib import Path
from remaining_table_replay import collect,validate_run


class TableReplayControls(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads(Path(__file__).with_name('remaining_table_protocol.json').read_text())
        self.rows=[{'lane':lane,'run':run,'seed':seed,'final':1.0+seed/10,
                    'curve':[[t,0.5,1.0+seed/10] for t in range(50,self.protocol['cases'][run]['steps']+1,50)],'train_seconds':1}
                   for lane in self.protocol['lanes'] for run in self.protocol['runs'] for seed in self.protocol['seeds']]

    def test_complete_seed_aggregation_is_sample_sd(self):
        cell=collect(self.rows,self.protocol)['torch291']['long-adamw-wsd']
        self.assertAlmostEqual(cell['mean'],1.1);self.assertAlmostEqual(cell['sample_sd'],0.1)
        self.assertEqual(cell['rounded_3dp'],['1.100','0.100'])

    def test_missing_and_duplicate_workers_rejected(self):
        with self.assertRaises(ValueError):collect(self.rows[:-1],self.protocol)
        with self.assertRaises(ValueError):collect(self.rows+[self.rows[0]],self.protocol)

    def test_missing_step_and_nonfinite_loss_rejected(self):
        row=copy.deepcopy(self.rows[0]);row['curve'].pop(0)
        with self.assertRaises(ValueError):validate_run(row,self.protocol)
        row=copy.deepcopy(self.rows[0]);row['curve'][0][2]=float('nan')
        with self.assertRaises(ValueError):validate_run(row,self.protocol)

    def test_final_mismatch_and_unplanned_seed_rejected(self):
        row=copy.deepcopy(self.rows[0]);row['final']=0.0
        with self.assertRaises(ValueError):validate_run(row,self.protocol)
        row=copy.deepcopy(self.rows[0]);row['seed']=100
        with self.assertRaises(ValueError):validate_run(row,self.protocol)

if __name__=='__main__':unittest.main()
