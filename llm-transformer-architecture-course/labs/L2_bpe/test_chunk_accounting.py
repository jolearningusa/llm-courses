import unittest

from bpe import ByteBPE
from chunk_accounting import dechunk_scalar, partition, position_ledger, prefix_entropy, report


class ChunkAccountingTests(unittest.TestCase):
    def test_entropy_uses_distribution_not_observed_byte_surprise(self):
        # Uniform distribution has 8 bits before any observation, regardless of byte.
        self.assertEqual(prefix_entropy(b'a')[1], [8.0])
        self.assertEqual(prefix_entropy(b'z')[1], [8.0])
        self.assertLess(prefix_entropy(b'a'*40)[1][-1], 8.0)

    def test_all_prefixes_and_changed_suffixes_preserve_boundary_decisions(self):
        raw = b'banana banana banana\n'
        flags, scores = prefix_entropy(raw)
        for cut in range(len(raw)+1):
            self.assertEqual(prefix_entropy(raw[:cut]), (flags[:cut], scores[:cut]))
            altered = prefix_entropy(raw[:cut] + bytes([255])*20)
            self.assertEqual(altered[0][:cut], flags[:cut])

    def test_future_sensitive_router_is_rejected_by_same_prefix_control(self):
        def wrong(data):
            return [True] + [data[-1] == 255]*(len(data)-1)
        a,b = b'abcd\x00', b'abcd\xff'
        self.assertEqual(a[:4], b[:4])
        self.assertNotEqual(wrong(a)[:4], wrong(b)[:4])
        self.assertEqual(prefix_entropy(a)[0][:4], prefix_entropy(b)[0][:4])

    def test_lossless_raw_bytes_and_split_utf8(self):
        for raw in (b'',bytes(range(256)),'缓存 🙂'.encode()):
            flags,_ = prefix_entropy(raw)
            parts = partition(raw, flags)
            self.assertEqual(b''.join(parts), raw)
            self.assertTrue(all(0<len(p)<=8 for p in parts))
        raw='🙂'.encode()
        parts=partition(raw,[True]*4)
        with self.assertRaises(UnicodeDecodeError): parts[0].decode('utf-8')
        self.assertEqual(b''.join(parts).decode('utf-8'),'🙂')

    def test_offline_bpe_is_lossless_but_not_incremental_byte_patching(self):
        bpe=ByteBPE().train('ab '*20,1)
        self.assertEqual(bpe.encode('a'),[97])
        self.assertEqual(bpe.encode('ab'),[256])
        self.assertEqual(bpe.decode([256]),'ab')

    def test_literal_causal_smoothing_and_forward_fill(self):
        self.assertEqual(dechunk_scalar(5,[0,3],[10,30],[1,.5]),[10,10,10,20,20])
        self.assertEqual(dechunk_scalar(5,[0,3],[10,999],[1,.5])[:3],[10,10,10])
        self.assertEqual(dechunk_scalar(3,[0],[10],[1]),[10,10,10])

    def test_full_patch_pooling_negative_control_leaks_future(self):
        # Broadcasting a whole-patch mean into its earlier positions reads future bytes.
        def wrong(values):return [sum(values)/len(values)]*len(values)
        self.assertNotEqual(wrong([10,20,30])[0],wrong([10,20,99])[0])

    def test_invalid_boundaries_or_configurations(self):
        for flags in ([False],[],[True,False]):
            with self.assertRaises(ValueError):partition(b'x',flags)
        for threshold,cap in [(float('nan'),8),(9,8),(7,0),(7,True)]:
            with self.assertRaises(ValueError):prefix_entropy(b'x',threshold,cap)
        with self.assertRaises(ValueError):dechunk_scalar(2,[1],[10],[1])

    def test_report_keeps_all_work_columns_and_equal_raw_input(self):
        data=report()
        self.assertEqual(len(data['rows']),12)
        for name in ('repeat','multilingual','code'):
            rows=[r for r in data['rows'] if r['sample']==name]
            self.assertEqual(len({r['input_sha256'] for r in rows}),1)
            for row in rows:
                self.assertTrue(row['round_trip'])
                self.assertEqual(sum(len(bytes.fromhex(p)) for p in row['piece_hex']),row['bytes'])
                if row['method']=='prefix_entropy_toy':
                    for field in ('local_encoder_positions','local_decoder_positions','router_positions'):
                        self.assertEqual(row['positions'][field],row['bytes'])
                self.assertEqual(row['positions']['model_execution'],'not_run')
        self.assertEqual(position_ledger(16,2,True,True)['global_positions'],2)


if __name__=='__main__':unittest.main()
