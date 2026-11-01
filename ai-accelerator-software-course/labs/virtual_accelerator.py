"""Deterministic CPU functional model, not a GPU/ISA/physical simulator.
Model ticks express event ordering only. No real MMIO, DMA, IRQ or firmware flash.
"""
from dataclasses import dataclass
import struct

ABI = 1
PACKET = struct.Struct('<BBHIIIII')


class ContractError(ValueError):
    pass


def uint(value, bits=32):
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ContractError('unsigned integer field out of range')
    return value


@dataclass(frozen=True)
class Descriptor:
    abi: int
    opcode: int
    reserved: int
    generation: int
    src: int
    dst: int
    length: int
    cookie: int

    def encode(self):
        values = (uint(self.abi, 8), uint(self.opcode, 8), uint(self.reserved, 16),
                  uint(self.generation), uint(self.src), uint(self.dst),
                  uint(self.length), uint(self.cookie))
        return PACKET.pack(*values)

    @classmethod
    def decode(cls, packet):
        if not isinstance(packet, bytes) or len(packet) != PACKET.size:
            raise ContractError('descriptor must be exactly 24 bytes')
        return cls(*PACKET.unpack(packet))


@dataclass
class Buffer:
    data: bytearray
    pins: int = 0


@dataclass
class Command:
    descriptor: Descriptor
    fault: str
    state: str = 'queued'
    observed: bool = False
    due: int = 0
    start: int = 0


class Device:
    """One FIFO worker, bounded outstanding slots, coherent modeled byte buffers.

    Pins persist through device completion until host reap. Reset is an atomic
    quiesce in this model only: old work cannot execute after it. Command history
    is retained for teaching, so this is not a production allocator/service.
    """
    def __init__(self, capacity=2, latency=2, watchdog=5, memory_limit=4096):
        if any(type(v) is not int or v <= 0 for v in
               (capacity, latency, watchdog, memory_limit)):
            raise ContractError('positive integer configuration required')
        self.capacity, self.latency, self.watchdog = capacity, latency, watchdog
        self.memory_limit = memory_limit
        self.state, self.generation, self.clock = 'RESET', 0, 0
        self.image = None
        self.buffers, self.commands, self.queue = {}, {}, []
        self.running = None
        self.next_buffer = 1
        self.trace = []
        self.stale_completions = 0

    def event(self, name, **fields):
        self.trace.append(dict(tick=self.clock, generation=self.generation,
                               event=name, **fields))

    def boot(self, image='tutorial-fw-v1', abi=ABI):
        if self.state != 'RESET':
            raise ContractError('boot requires RESET')
        if image != 'tutorial-fw-v1' or type(abi) is not int or abi != ABI:
            self.state = 'FAULT'
            self.event('boot_rejected')
            raise ContractError('unsupported image/ABI')
        self.image, self.state = image, 'READY'
        self.event('boot_ready', image=image)

    def alloc(self, size):
        uint(size)
        if not size or sum(len(b.data) for b in self.buffers.values()) + size > self.memory_limit:
            raise ContractError('allocation limit')
        handle = self.next_buffer
        self.next_buffer += 1
        self.buffers[handle] = Buffer(bytearray(size))
        self.event('alloc', buffer=handle, size=size)
        return handle

    def buffer(self, handle):
        uint(handle)
        if handle not in self.buffers:
            raise ContractError('unknown buffer handle')
        return self.buffers[handle]

    def write(self, handle, data):
        b = self.buffer(handle)
        if b.pins:
            raise ContractError('host write while buffer pinned')
        if not isinstance(data, bytes) or len(data) != len(b.data):
            raise ContractError('exact bytes buffer write required')
        b.data[:] = data

    def read(self, handle):
        b = self.buffer(handle)
        if b.pins:
            raise ContractError('reap completion before host read')
        return bytes(b.data)

    def free(self, handle):
        b = self.buffer(handle)
        if b.pins:
            raise ContractError('free while device/host completion owns buffer')
        del self.buffers[handle]
        self.event('free', buffer=handle)

    def submit(self, packet, fault='none'):
        if self.state != 'READY':
            raise ContractError('device not ready')
        d = Descriptor.decode(packet)
        if d.abi != ABI or d.opcode not in (1, 2) or d.reserved != 0:
            raise ContractError('unsupported ABI/opcode/reserved bits')
        if d.generation != self.generation or not d.cookie or not d.length:
            raise ContractError('stale generation or empty field')
        src, dst = self.buffer(d.src), self.buffer(d.dst)
        if d.src == d.dst or d.length > min(len(src.data), len(dst.data)):
            raise ContractError('alias or bounds violation')
        if src.pins or dst.pins:
            raise ContractError('exclusive model buffer ownership')
        key = (d.generation, d.cookie)
        if key in self.commands:
            raise ContractError('duplicate cookie in generation')
        if fault not in ('none', 'drop_irq', 'dma_error', 'hang'):
            raise ContractError('unknown fault')
        active = sum(c.state not in ('reaped', 'reset') for c in self.commands.values())
        if active >= self.capacity:
            raise ContractError('queue full until completion reaped')
        # No resource mutation occurs before all checks pass.
        src.pins += 1
        dst.pins += 1
        self.commands[key] = Command(d, fault)
        self.queue.append(key)
        self.event('submit', cookie=d.cookie)
        return key

    def command(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ContractError('invalid ticket')
        uint(key[0]); uint(key[1])
        if key[0] != self.generation:
            raise ContractError('stale ticket')
        if key not in self.commands:
            raise ContractError('unknown ticket')
        return self.commands[key]

    def deliver_completion(self, key, error=False, drop_irq=False):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ContractError('invalid completion ticket')
        uint(key[0]); uint(key[1])
        if key[0] != self.generation:
            self.stale_completions += 1
            self.event('stale_completion_rejected')
            return False
        c = self.command(key)
        if c.state != 'inflight' or key != self.running:
            raise ContractError('duplicate/out-of-order completion')
        d = c.descriptor
        # Atomic functional write precedes the modeled completion publication.
        if not error:
            data = bytes(self.buffers[d.src].data[:d.length])
            if d.opcode == 2:
                data = bytes((x + 1) % 256 for x in data)
            self.buffers[d.dst].data[:d.length] = data
            self.event('output_visible', cookie=d.cookie)
        c.state = 'error' if error else 'done'
        c.observed = not drop_irq
        self.running = None
        self.event('completion', cookie=d.cookie, status=c.state, irq=not drop_irq)
        return True

    def tick(self):
        self.clock += 1
        if self.state != 'READY':
            return
        if self.running is None and self.queue:
            self.running = self.queue.pop(0)
            c = self.commands[self.running]
            c.state, c.start, c.due = 'inflight', self.clock, self.clock + self.latency
            self.event('start', cookie=c.descriptor.cookie)
        if self.running is not None:
            c = self.commands[self.running]
            if self.clock - c.start >= self.watchdog:
                self.state = 'FAULT'
                self.event('watchdog_fault')
            elif c.fault != 'hang' and self.clock >= c.due:
                self.deliver_completion(self.running, error=c.fault == 'dma_error',
                                        drop_irq=c.fault == 'drop_irq')

    def poll(self, key):
        c = self.command(key)
        if c.state in ('done', 'error', 'cancelled'):
            c.observed = True
        return c.state

    def wait(self, key, timeout=5, polling=False):
        uint(timeout)
        for _ in range(timeout + 1):
            c = self.command(key)
            if c.state in ('done', 'error', 'cancelled') and (c.observed or polling):
                c.observed = True
                return c.state
            if _ < timeout:
                self.tick()
        raise ContractError('timeout; ownership remains pinned')

    def cancel(self, key):
        c = self.command(key)
        if c.state != 'queued':
            raise ContractError('only queued work is cancellable in this model')
        self.queue.remove(key)
        c.state, c.observed = 'cancelled', True
        self.event('cancel', cookie=c.descriptor.cookie)

    def reap(self, key):
        c = self.command(key)
        if c.state not in ('done', 'error', 'cancelled') or not c.observed:
            raise ContractError('completion not safely observed')
        result = c.state
        for handle in (c.descriptor.src, c.descriptor.dst):
            self.buffers[handle].pins -= 1
        c.state = 'reaped'
        self.event('reap', cookie=c.descriptor.cookie, status=result)
        return result

    def reset(self):
        # This method models successful quiescence; it is not a hardware reset driver.
        for c in self.commands.values():
            if c.state != 'reaped':
                c.state = 'reset'
        for b in self.buffers.values():
            b.pins = 0
        self.queue.clear()
        self.running = None
        self.generation += 1
        self.state, self.image = 'RESET', None
        self.event('quiesced_reset')

    def registers(self):
        return dict(status=self.state, generation=self.generation, tick=self.clock,
                    queued=len(self.queue), running=self.running,
                    allocated_bytes=sum(len(b.data) for b in self.buffers.values()),
                    stale_completions=self.stale_completions)

    def assert_invariants(self):
        expected = {h: 0 for h in self.buffers}
        for key, c in self.commands.items():
            if c.state not in ('reaped', 'reset'):
                assert key[0] == self.generation
                for h in (c.descriptor.src, c.descriptor.dst):
                    assert h in expected
                    expected[h] += 1
        assert all(self.buffers[h].pins == v and v in (0, 1) for h, v in expected.items())
        assert all(self.commands[k].state == 'queued' for k in self.queue)
        assert self.running is None or self.commands[self.running].state == 'inflight'
        assert len(self.queue) == len(set(self.queue))
        return True
