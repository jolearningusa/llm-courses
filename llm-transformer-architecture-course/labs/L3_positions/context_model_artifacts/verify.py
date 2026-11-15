"""Independent tokenizer/scorer replay of retained pilot outputs; no model load/generation."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    here=Path(__file__).resolve().parent
    plan=json.loads((here/'plan.json').read_text()); result=json.loads((here/'results.json').read_text())
    assert hashlib.sha256((here/'plan.json').read_bytes()).hexdigest()==result['plan_sha256']
    for name,h in plan['sources'].items():assert hashlib.sha256((here.parent/name).read_bytes()).hexdigest()==h
    for name,h in plan['model_files'].items():assert hashlib.sha256((a.model_dir/name).read_bytes()).hexdigest()==h
    assert plan['sources']==result['sources']
    assert len(plan['rows'])==len(result['rows'])==90
    assert len({r['case']['id'] for r in result['rows']})==90
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False)
    correct=completed=cap_hits=0
    cells={}; ranges={}; positions={}
    for index,(f,r) in enumerate(zip(plan['rows'],result['rows'])):
        assert all(r[k]==v for k,v in f.items());assert r['execution_index']==index
        tokenized=tokenizer(r['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        assert digest(tokenized['input_ids'])==r['input_ids_sha256']
        n=len(tokenized['input_ids']);assert n==r['input_tokens']
        assert r['eligible']==(n<=4096 and n+24<=plan['configured_window'])
        for span in r['evidence_spans']:
            lo,hi=span['char_start'],span['char_end']
            assert r['prompt'][lo:hi]==r['case']['records'][span['record_index']]
            indices=[i for i,(start,end) in enumerate(tokenized['offset_mapping']) if start<hi and end>lo]
            assert (min(indices),max(indices)+1)==(span['token_start'],span['token_end_exclusive'])
        c=r['case'];key=f"{c['length_records']}:{c['position']}:{c['task']}"
        cell=cells.setdefault(key,dict(planned=0,eligible=0,completed=0,correct=0,cap_hits=0,errors=0))
        cell['planned']+=1;cell['eligible']+=r['eligible'];cell['errors']+=r['status']=='error'
        if r['status']=='completed':
            text=tokenizer.decode(r['output_ids'],skip_special_tokens=True);assert text==r['text']
            score=text.strip()==c['expected_answer'];assert score==r['correct']
            cell['completed']+=1;cell['correct']+=score;cell['cap_hits']+=r['hit_output_cap_without_eos']
            completed+=1;correct+=score;cap_hits+=r['hit_output_cap_without_eos']
        ranges.setdefault(str(c['length_records']) if c['position']!='evidence-only' else 'evidence-only',[]).append(n)
        positions.setdefault(c['position'],[]).append(r['evidence_spans'][0]['token_start']/n)
    for key,value in cells.items():assert all(result['cells'][key][k]==v for k,v in value.items())
    report={'scope':'tokenizer/scorer replay only; no regenerated responses','source_revision':result['source_revision'],
            'results_sha256':hashlib.sha256((here/'results.json').read_bytes()).hexdigest(),
            'plan_sha256':result['plan_sha256'],'verified_cases':90,'completed':completed,'strict_correct':correct,'cap_hits':cap_hits,
            'input_token_ranges':{k:[min(v),max(v)] for k,v in ranges.items()},
            'first_evidence_start_fraction_of_complete_input':{k:[min(v),max(v)] for k,v in positions.items()},'cells':cells}
    with a.output.open('x') as out:json.dump(report,out,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))

if __name__=='__main__':main()
