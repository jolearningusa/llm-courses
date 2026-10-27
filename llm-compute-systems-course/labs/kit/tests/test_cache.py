import unittest
from cache_lab import Identity,PrefixCache
class CacheTests(unittest.TestCase):
    def test_prefix_not_suffix_and_partial_block(self):
        c=PrefixCache(8,4);i=Identity('v1')
        self.assertEqual(c.acquire('a',list(range(9)),i)['recomputed_tokens'],9);c.release('a')
        self.assertEqual(c.acquire('b',list(range(9)),i)['matched_tokens'],8);c.release('b')
        self.assertEqual(c.acquire('c',[99]+list(range(1,9)),i)['matched_tokens'],0);c.release('c')
        self.assertEqual(c.acquire('d',list(range(4))+[99,5,6,7],i)['matched_tokens'],4)
    def test_identity_and_shared_release(self):
        c=PrefixCache(8);i=Identity('v1');t=list(range(8))
        c.acquire('a',t,i);c.acquire('b',t,i)
        self.assertEqual(c.snapshot()['references'],4)
        c.release('a');c.release('a')
        self.assertEqual(c.snapshot()['pinned_blocks'],2)
        self.assertEqual(c.acquire('c',t,Identity('v2'))['matched_tokens'],0)
        self.assertEqual(c.acquire('d',t,Identity('v1',sharing_domain='other'))['matched_tokens'],0)
    def test_pressure_and_rollback(self):
        c=PrefixCache(2);i=Identity('v1')
        c.acquire('a',list(range(8)),i)
        with self.assertRaises(MemoryError):c.acquire('b',list(range(12)),i)
        self.assertEqual(c.snapshot()['references'],2)
        c.release('a')
        c.acquire('c',list(range(20,28)),i);c.release('c')
        self.assertEqual(c.snapshot()['references'],0)
        self.assertEqual(c.acquire('d',list(range(8)),i)['matched_tokens'],0)
