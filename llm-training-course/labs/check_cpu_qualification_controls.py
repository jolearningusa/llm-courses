"""Exercise legal and false-pass controls; write results to a new JSON path."""
import sys
import importlib.util,json,tempfile,contextlib,io
from pathlib import Path
from unittest.mock import patch
source=Path(__file__).resolve().with_name('cpu_qualification.py')
spec=importlib.util.spec_from_file_location('qualification',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if len(sys.argv) != 2 or Path(sys.argv[1]).exists():
 raise SystemExit('Usage: check_cpu_qualification_controls.py NEW_OUTPUT.json')
rows=[]
with tempfile.TemporaryDirectory(prefix='trn4-controls-') as root:
 root=Path(root)
 for name,text,expected in [('legal','import unittest\nclass TestFixture(unittest.TestCase):\n def test_literal(self): self.assertEqual(2 + 2, 4)\n',0),('empty','',1),('failure','import unittest\nclass TestFixture(unittest.TestCase):\n def test_literal(self): self.assertEqual(2 + 2, 5)\n',1),('skip','import unittest\nclass TestFixture(unittest.TestCase):\n @unittest.skip("fixture")\n def test_literal(self): pass\n',1)]:
  folder=root/name;folder.mkdir();(folder/f'test_{name}.py').write_text(text)
  log=io.StringIO()
  with contextlib.redirect_stderr(log): result=m.worker(folder,root/(name+'.json'))
  assert result==expected
  row=json.loads((root/(name+'.json')).read_text())
  rows.append({'case':name,'expected_exit':expected,'actual_exit':result,'record':row,'log':log.getvalue()})
 with patch.object(m.metadata,'version',lambda name:'0.0.0' if name=='torch' else __import__('importlib').metadata.distribution(name).version):
  try: m.qualify(root/'drift')
  except RuntimeError as e:
   assert 'versions differ' in str(e); rows.append({'case':'version-drift','rejected':True,'message':str(e),'output_created':(root/'drift').exists()})
  else: raise AssertionError('Drift accepted')
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+'\n')
print('Five runner controls passed: legal, empty, failure, skip, version drift')
