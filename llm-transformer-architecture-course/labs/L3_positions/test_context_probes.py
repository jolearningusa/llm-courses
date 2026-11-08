import copy
import unittest
from context_probes import build_case, evidence, manifest, prompt_for, score_answer, solve_fixture, summary


class ContextProbes(unittest.TestCase):
    def test_grid_and_debug_oracles(self):
        result = summary(manifest())
        self.assertEqual((result['case_count'], result['cells'], result['oracle_agreement']), (81, 27, 81))
        self.assertEqual(result['model_runs'], 0)

    def test_move_evidence_without_changing_filler_or_question(self):
        for task in ('lookup', 'chain', 'count'):
            cases = [build_case(32, p, task, replica=1) for p in ('early', 'middle', 'late')]
            fillers = [[s for s in c['records'] if s.startswith('DISTRACTOR')] for c in cases]
            self.assertEqual(fillers[0], fillers[1]); self.assertEqual(fillers[1], fillers[2])
            self.assertEqual(len({c['question'] for c in cases}), 1)
            self.assertEqual(len({c['expected_answer'] for c in cases}), 1)
            self.assertEqual(len({c['evidence_indices'][0] for c in cases}), 3)

    def test_lengths_preserve_task_and_answer_with_valid_indices(self):
        for task in ('lookup', 'chain', 'count'):
            cases = [build_case(n, 'late', task) for n in (32,64,128)]
            self.assertEqual(len({c['expected_answer'] for c in cases}), 1)
            for c in cases:
                self.assertEqual(len(c['records']), c['length_records'])
                self.assertTrue(all(0 <= i < len(c['records']) for i in c['evidence_indices']))

    def test_nontrivial_replicas_and_repeatability(self):
        self.assertEqual(manifest(), manifest())
        self.assertEqual([evidence('count',7,r)[3] for r in range(3)], ['2','3','4'])
        self.assertNotEqual(evidence('lookup',7,0)[3], evidence('lookup',7,1)[3])

    def test_literal_chain_oracle_and_broken_evidence(self):
        c = {'task':'chain','target':'a','records':['VALUE c result','LINK b c','LINK a b'],
             'expected_answer':'result'}
        self.assertEqual(solve_fixture(c), 'result')
        bad = copy.deepcopy(c); bad['records'][0] = 'VALUE c wrong'
        self.assertFalse(score_answer(c, solve_fixture(bad)))

    def test_scoring_and_prompt_boundary(self):
        c = build_case(32, 'middle', 'lookup')
        self.assertTrue(score_answer(c, c['expected_answer']+'\n'))
        self.assertFalse(score_answer(c, 'wrong')); self.assertFalse(score_answer(c, None))
        self.assertNotIn('expected_answer', prompt_for(c)); self.assertNotIn(c['id'], prompt_for(c))
        self.assertTrue(prompt_for(c).endswith(c['question']))

    def test_invalid_configuration(self):
        for n in (True, 3, 8.5):
            with self.assertRaises(ValueError): build_case(n, 'early', 'lookup')
        with self.assertRaises(ValueError): build_case(32, 'unknown', 'lookup')
        with self.assertRaises(ValueError): manifest(replicas=0)


if __name__ == '__main__': unittest.main()
