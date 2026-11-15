import copy
import unittest
from context_model_pilot import cases, messages, evidence_spans, eligible, validate_frozen, summarize
from context_probes import score_answer
import json
from pathlib import Path


class ModelPilotControls(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path(__file__).with_name('context_model_protocol.json').read_text())

    def test_full_grid_and_short_controls(self):
        data = cases(self.protocol)
        self.assertEqual(len(data), 90)
        self.assertEqual(len({c['id'] for c in data}), 90)
        self.assertEqual(sum(c['position'] == 'evidence-only' for c in data), 9)
        for c in data[:9]:
            self.assertEqual(c['evidence_indices'], list(range(len(c['records']))))

    def test_prompt_does_not_accept_labels_or_case_metadata(self):
        case = cases(self.protocol)[0]
        prompt = messages(case['records'], case['question'], self.protocol['system'])
        mutated = copy.deepcopy(case);mutated.update(expected_answer='POISON', id='POISON', target='POISON')
        self.assertEqual(prompt, messages(mutated['records'], mutated['question'], self.protocol['system']))
        self.assertNotIn('POISON', str(prompt))

    def test_evidence_offsets_include_partial_token_overlap(self):
        prompt = 'xxContext:\nAAA\nBBB\nQuestion: q'
        offsets = [(0,11),(11,13),(13,16),(16,19)]
        spans = evidence_spans(prompt, ['AAA','BBB'], [1], offsets)
        self.assertEqual((spans[0]['char_start'], spans[0]['char_end']), (15,18))
        self.assertEqual((spans[0]['token_start'],spans[0]['token_end_exclusive']), (2,4))
        with self.assertRaises(ValueError): evidence_spans(prompt, ['AAA','BBB'], [1], [])

    def test_caps_and_generation_reserve(self):
        self.assertTrue(eligible(4096, self.protocol, 8192))
        self.assertFalse(eligible(4097, self.protocol, 8192))
        self.assertFalse(eligible(4096, self.protocol, 4100))

    def test_mutated_frozen_plan_rejected(self):
        validate_frozen({'x':[1]}, {'x':[1]})
        with self.assertRaises(ValueError): validate_frozen({'x':[2]}, {'x':[1]})

    def test_strict_score_preserves_prose_failures(self):
        self.assertTrue(score_answer({'expected_answer':'2'}, ' 2\n'))
        self.assertFalse(score_answer({'expected_answer':'2'}, 'There are 2.'))
        self.assertFalse(score_answer({'expected_answer':'v123'}, 'not v123'))

    def test_errors_ineligible_and_cap_hits_stay_in_denominator(self):
        c={'length_records':32,'position':'early','task':'lookup'}
        rows=[{'case':c,'input_tokens':100,'eligible':True,'status':'completed','correct':False,
               'output_ids':[1,2], 'hit_output_cap_without_eos':True},
              {'case':c,'input_tokens':100,'eligible':True,'status':'error'},
              {'case':c,'input_tokens':5000,'eligible':False,'status':'ineligible'}]
        cell=summarize(rows)['32:early:lookup']
        self.assertEqual([cell[k] for k in ['planned','eligible','completed','correct','errors','cap_hits']], [3,2,1,0,1,1])

if __name__ == '__main__': unittest.main()
