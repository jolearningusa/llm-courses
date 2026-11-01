import unittest
from dataclasses import replace
from experiments import explore, fuzz, gate, report, scenarios, setup
from virtual_accelerator import ABI, PACKET, ContractError, Descriptor, Device


class DeviceTests(unittest.TestCase):
    def blank(self, **kw):
        d = Device(**kw); d.boot(); a = d.alloc(4); b = d.alloc(4)
        return d, a, b

    def desc(self, d, a, b, **kw):
        return replace(Descriptor(ABI, 1, 0, d.generation, a, b, 4, 1), **kw)

    def test_packet_roundtrip_and_size(self):
        d, a, b = self.blank(); packet = self.desc(d, a, b).encode()
        self.assertEqual(PACKET.size, 24)
        self.assertEqual(Descriptor.decode(packet), self.desc(d, a, b))

    def test_malformed_packet_and_types(self):
        for value in [b'', bytes(23), bytes(25), bytearray(24), None]:
            with self.assertRaises(ContractError): Descriptor.decode(value)
        with self.assertRaises(ContractError): Descriptor(True, 1, 0, 0, 1, 2, 4, 1).encode()
        with self.assertRaises(ContractError): Descriptor(1, 1, 0, 0, 1, 2, -1, 1).encode()

    def test_boot_contract(self):
        d = Device()
        with self.assertRaises(ContractError): d.boot(abi=9)
        self.assertEqual(d.state, 'FAULT')
        d.reset(); d.boot()
        with self.assertRaises(ContractError): d.boot()

    def test_not_ready_rejects_submission(self):
        d = Device(); a = d.alloc(4); b = d.alloc(4)
        with self.assertRaises(ContractError): d.submit(self.desc(d, a, b).encode())
        self.assertEqual(d.buffers[a].pins, 0)

    def test_all_invalid_descriptors_leave_resources_unchanged(self):
        d, a, b = self.blank()
        for kw in [{'abi': 2}, {'opcode': 99}, {'reserved': 1}, {'generation': 4},
                   {'src': 99}, {'src': b}, {'length': 5}, {'length': 0}, {'cookie': 0}]:
            with self.subTest(kw=kw):
                with self.assertRaises(ContractError): d.submit(self.desc(d, a, b, **kw).encode())
                self.assertEqual(d.queue, []); self.assertEqual(d.commands, {})
                self.assertEqual(d.buffers[a].pins, 0)

    def test_copy_and_write_before_completion(self):
        d, a, b, key = setup()
        self.assertEqual(bytes(d.buffers[b].data), bytes(4))
        d.wait(key); d.reap(key)
        self.assertEqual(d.read(b), bytes([0, 1, 254, 255]))
        names = [e['event'] for e in d.trace]
        self.assertLess(names.index('output_visible'), names.index('completion'))

    def test_add_wraparound(self):
        d, a, b, key = setup(opcode=2); d.wait(key); d.reap(key)
        self.assertEqual(d.read(b), bytes([1, 2, 255, 0]))

    def test_partial_copy_preserves_destination_tail(self):
        d, a, b = self.blank(); d.write(a, bytes([1, 2, 3, 4])); d.write(b, b'xxxx')
        k = d.submit(self.desc(d, a, b, length=2).encode()); d.wait(k); d.reap(k)
        self.assertEqual(d.read(b), b'\x01\x02xx')

    def test_host_access_and_free_while_pinned(self):
        d, a, b, key = setup()
        for action in [lambda: d.write(a, b'abcd'), lambda: d.read(b), lambda: d.free(a)]:
            with self.assertRaises(ContractError): action()
        d.wait(key)
        with self.assertRaises(ContractError): d.free(a)
        d.reap(key); d.free(a); d.free(b)
        self.assertEqual(d.registers()['allocated_bytes'], 0)

    def test_timeout_does_not_unpin(self):
        d, a, b, key = setup()
        with self.assertRaises(ContractError): d.wait(key, timeout=1)
        self.assertEqual(d.buffers[a].pins, 1)
        d.wait(key); d.reap(key); d.assert_invariants()

    def test_queued_cancel_then_reap(self):
        d, a, b, key = setup(); d.cancel(key)
        self.assertEqual(d.reap(key), 'cancelled')
        for _ in range(5): d.tick()
        self.assertEqual(d.read(b), bytes(4)); d.assert_invariants()

    def test_inflight_cancel_is_not_supported(self):
        d, a, b, key = setup(); d.tick()
        with self.assertRaises(ContractError): d.cancel(key)
        self.assertEqual(d.buffers[a].pins, 1)

    def test_lost_interrupt_polling_recovers(self):
        d, a, b, key = setup('drop_irq')
        with self.assertRaises(ContractError): d.wait(key, timeout=4)
        self.assertEqual(d.command(key).state, 'done')
        with self.assertRaises(ContractError): d.reap(key)
        self.assertEqual(d.poll(key), 'done'); d.reap(key)
        self.assertEqual(d.read(b), bytes([0, 1, 254, 255]))

    def test_dma_error_does_not_write(self):
        d, a, b, key = setup('dma_error')
        self.assertEqual(d.wait(key), 'error'); self.assertEqual(d.reap(key), 'error')
        self.assertEqual(d.read(b), bytes(4))

    def test_watchdog_requires_reset(self):
        d, a, b, key = setup('hang')
        with self.assertRaises(ContractError): d.wait(key, timeout=8)
        self.assertEqual(d.state, 'FAULT'); self.assertEqual(d.buffers[a].pins, 1)
        d.reset(); d.boot(); self.assertEqual(d.buffers[a].pins, 0)

    def test_reset_stale_and_generation_reuse(self):
        d, a, b, old = setup(); d.tick(); d.reset()
        self.assertFalse(d.deliver_completion(old)); self.assertEqual(d.stale_completions, 1)
        with self.assertRaises(ContractError): d.reap(old)
        d.boot(); new = d.submit(self.desc(d, a, b).encode())
        self.assertNotEqual(old, new); d.wait(new); d.reap(new); d.assert_invariants()

    def test_double_completion_and_reap(self):
        d, a, b, key = setup(); d.wait(key)
        with self.assertRaises(ContractError): d.deliver_completion(key)
        d.reap(key)
        with self.assertRaises(ContractError): d.reap(key)

    def test_queue_limit_until_reaped(self):
        d, a, b = self.blank(capacity=1); k=d.submit(self.desc(d, a, b).encode()); d.wait(k)
        x,y = d.alloc(4),d.alloc(4)
        with self.assertRaises(ContractError): d.submit(self.desc(d, x, y, cookie=2).encode())
        d.reap(k); d.submit(self.desc(d, x, y, cookie=2).encode()); d.assert_invariants()

    def test_fifo_and_exclusive_buffers(self):
        d, a, b, key = setup(); x,y=d.alloc(4),d.alloc(4)
        with self.assertRaises(ContractError): d.submit(self.desc(d, a, y, cookie=2).encode())
        second=d.submit(self.desc(d,x,y,cookie=2).encode());d.wait(second,timeout=8)
        ends=[e['cookie'] for e in d.trace if e['event']=='completion']
        self.assertEqual(ends,[1,2]);d.assert_invariants()

    def test_allocation_and_unknown_fault_failure(self):
        d,a,b=self.blank(memory_limit=8)
        with self.assertRaises(ContractError):d.alloc(1)
        with self.assertRaises(ContractError):d.submit(self.desc(d,a,b).encode(),fault='bad')
        self.assertEqual(d.buffers[a].pins,0)

    def test_bad_tickets_and_configuration(self):
        d,a,b=self.blank()
        for key in [None, (), (0, []), (False, 1), (0, 99)]:
            with self.assertRaises(ContractError):d.poll(key)
        for kw in [{'capacity':0},{'latency':True},{'watchdog':-1}]:
            with self.assertRaises(ContractError):Device(**kw)

    def test_seeded_state_exercise(self):
        for seed in range(20):
            r=fuzz(seed,300); self.assertEqual(r['accepted']+r['rejected'],300)
            self.assertGreater(r['accepted'],0)
        self.assertEqual(fuzz(17,100),fuzz(17,100))

    def test_bounded_search_has_failing_control(self):
        self.assertIsNone(explore(False)['counterexample'])
        self.assertEqual(explore(True)['counterexample'],['submit','release'])
        self.assertIsNone(explore(True,depth=1)['counterexample'])
        with self.assertRaises(ValueError): explore(depth=-1)

    def test_report_and_scope_gate(self):
        self.assertEqual(len(scenarios()),5)
        r=report();self.assertEqual(r['time_unit'],'model_tick')
        self.assertTrue(gate(r)['accepted']);self.assertFalse(gate(r,'hardware')['accepted'])
        self.assertFalse(gate({'execution':'cpu-functional-model','checks':{}})['accepted'])
        self.assertFalse(gate({'execution':'cpu-functional-model','checks':{'unrelated':'passed'}})['accepted'])
        r['checks']['scenarios']='failed';self.assertFalse(gate(r)['accepted'])


if __name__ == '__main__':unittest.main()
