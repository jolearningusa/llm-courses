// Bounded Metal shared-buffer pilot; logical cancellation always drains submitted work.
import Foundation
import Metal
import CryptoKit

struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure(message) }
}
func digest(_ url: URL) throws -> String {
    SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
}
func statusName(_ status: MTLCommandBufferStatus) -> String {
    switch status {
    case .notEnqueued: return "notEnqueued"
    case .enqueued: return "enqueued"
    case .committed: return "committed"
    case .scheduled: return "scheduled"
    case .completed: return "completed"
    case .error: return "error"
    @unknown default: return "unknown"
    }
}
final class CompletionRecord: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    private var final = "none"
    func observe(_ status: MTLCommandBufferStatus) {
        lock.lock(); defer { lock.unlock() }
        count += 1; final = statusName(status)
    }
    func snapshot() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return ["callbacks": count, "status": final]
    }
}
final class Slot {
    let input: any MTLBuffer
    let output: any MTLBuffer
    let coverage: any MTLBuffer
    let capacity: Int
    private(set) var state = "available"
    private(set) var cancelled = false
    private(set) var n = 0
    private(set) var run = 0
    init(device: any MTLDevice, capacity: Int) throws {
        self.capacity = capacity
        guard let a = device.makeBuffer(length: capacity * 4, options: .storageModeShared),
              let b = device.makeBuffer(length: capacity * 4, options: .storageModeShared),
              let c = device.makeBuffer(length: 4, options: .storageModeShared) else {
            throw Failure("buffer allocation failed")
        }
        input = a; output = b; coverage = c
    }
    func fill(count: Int, run: Int) throws {
        try require(state == "available", "host write requires available ownership")
        try require(count > 0 && count <= capacity, "invalid element count")
        n = count; self.run = run; cancelled = false
        let a = input.contents().bindMemory(to: UInt32.self, capacity: capacity)
        let b = output.contents().bindMemory(to: UInt32.self, capacity: capacity)
        for i in 0..<capacity { a[i] = UInt32(i * 17 + run * 19); b[i] = 0xDEADBEEF }
        coverage.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        state = "prepared"
    }
    func markSubmitted() throws {
        try require(state == "prepared" && !cancelled, "submit requires uncancelled prepared state")
        state = "pending"
    }
    func cancel() throws {
        try require(state == "prepared" || state == "pending", "cancel requires prepared or pending state")
        cancelled = true
    }
    func quiesced() throws {
        try require(state == "pending", "quiesce requires pending work")
        state = "quiesced"
    }
    func verifyPhysicalResult() throws -> [String: Any] {
        try require(state == "quiesced", "host read requires drained GPU work")
        let b = output.contents().bindMemory(to: UInt32.self, capacity: capacity)
        var values: [UInt32] = []
        for i in 0..<capacity {
            let expected: UInt32 = i < n ? (UInt32(i * 17 + run * 19) ^ 0xA5A5A5A5) &+ UInt32(i) : 0xDEADBEEF
            try require(b[i] == expected, "output or untouched tail mismatch at \(i)")
            if i < n { values.append(b[i]) }
        }
        let observed = coverage.contents().load(as: UInt32.self)
        try require(observed == UInt32(n), "kernel coverage mismatch")
        return ["observed_valid_threads": observed, "output": values,
                "untouched_tail_elements": capacity - n, "tail_verified": true]
    }
    func publish() throws {
        try require(state == "quiesced" && !cancelled, "publication requires uncancelled drained work")
    }
    func recycle() throws {
        try require(state == "quiesced" || (state == "prepared" && cancelled), "recycle requires quiescence or never-submitted cancellation")
        state = "available"
    }
}

func main() throws {
    guard CommandLine.arguments.count == 6 else {
        throw Failure("usage: binary run_id shader.metal protocol.json runner.swift output.json")
    }
    guard let run = Int(CommandLine.arguments[1]), (1...3).contains(run) else { throw Failure("invalid run") }
    let shaderURL = URL(fileURLWithPath: CommandLine.arguments[2])
    let protocolURL = URL(fileURLWithPath: CommandLine.arguments[3])
    let runnerURL = URL(fileURLWithPath: CommandLine.arguments[4])
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[5])
    try require(!FileManager.default.fileExists(atPath: outputURL.path), "preserve prior output")
    let protocolData = try JSONSerialization.jsonObject(with: Data(contentsOf: protocolURL)) as! [String: Any]
    let capacity = protocolData["capacity_elements"] as! Int
    let baseline = protocolData["baseline_sizes"] as! [Int]
    let width = protocolData["threadgroup_width"] as! Int
    guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("not_run: no Metal device") }
    try require(device.hasUnifiedMemory, "unsupported: this fixture requires unified shared memory")
    guard let queue = device.makeCommandQueue(), let event = device.makeSharedEvent() else {
        throw Failure("unsupported: queue/shared event unavailable")
    }
    let buildStart = ProcessInfo.processInfo.systemUptime
    let library = try device.makeLibrary(source: String(contentsOf: shaderURL, encoding: .utf8), options: nil)
    guard let function = library.makeFunction(name: "lifetime_probe") else { throw Failure("missing intended kernel") }
    let pipeline = try device.makeComputePipelineState(function: function)
    try require(width <= pipeline.maxTotalThreadsPerThreadgroup, "unsupported threadgroup width")
    let compileSeconds = ProcessInfo.processInfo.systemUptime - buildStart
    let slot = try Slot(device: device, capacity: capacity)
    var cases: [[String: Any]] = []
    var guards: [[String: Any]] = []
    var committed = 0
    var callbacks = 0
    func rejection(_ label: String, _ action: () throws -> Void) throws {
        do { try action() }
        catch let error as Failure { guards.append(["control": label, "rejected": true, "reason":error.description]); return }
        throw Failure("guard accepted forbidden operation: \(label)")
    }
    func execute(_ label: String, _ n: Int, holdAndCancel: Bool = false) throws {
        try slot.fill(count: n, run: run)
        let descriptor = MTLCommandBufferDescriptor(); descriptor.retainedReferences = true
        guard let command = queue.makeCommandBuffer(descriptor: descriptor) else { throw Failure("no command buffer") }
        command.label = "course-\(run)-\(label)"
        let observed = CompletionRecord()
        command.addCompletedHandler { completed in observed.observe(completed.status) }
        if holdAndCancel { command.encodeWaitForEvent(event, value: 1) }
        guard let encoder = command.makeComputeCommandEncoder() else { throw Failure("no compute encoder") }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(slot.input, offset: 0, index: 0)
        encoder.setBuffer(slot.output, offset: 0, index: 1)
        encoder.setBuffer(slot.coverage, offset: 0, index: 2)
        var count = UInt32(n); encoder.setBytes(&count, length: 4, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: (n + width - 1) / width, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        try slot.markSubmitted()
        let tick = ProcessInfo.processInfo.systemUptime
        command.commit(); committed += 1
        // Even a guard failure must release the gate and drain before dropping resources.
        defer { if holdAndCancel { event.signaledValue = 1 }; command.waitUntilCompleted() }
        var pending: [String: Any] = [:]
        if holdAndCancel {
            pending = ["status": statusName(command.status), "event_value":event.signaledValue,
                       "completion": observed.snapshot(), "owner_state":slot.state]
            try require(event.signaledValue == 0 && command.status != .completed && command.status != .error,
                        "deterministic pending condition absent")
            try rejection("read_pending") { _ = try slot.verifyPhysicalResult() }
            try rejection("write_pending") { try slot.fill(count: n, run: run) }
            try rejection("recycle_pending") { try slot.recycle() }
            try rejection("resubmit_pending") { try slot.markSubmitted() }
            try slot.cancel()
            try require(slot.state == "pending", "logical cancellation must retain pending ownership")
            event.signaledValue = 1
        }
        command.waitUntilCompleted()
        try require(command.status == .completed && command.error == nil, "command failed: \(String(describing:command.error))")
        try require(observed.snapshot()["callbacks"] as? Int == 1, "completion callback count mismatch")
        callbacks += 1
        try slot.quiesced()
        let physical = try slot.verifyPhysicalResult()
        if holdAndCancel { try rejection("publish_cancelled") { try slot.publish() } }
        else { try slot.publish() }
        cases.append(["case":label,"n":n,"dispatch_lanes":((n+width-1)/width)*width,
                      "command_status":statusName(command.status),"completion":observed.snapshot(),
                      "retained_references":command.retainedReferences,"storage_mode":"shared",
                      "logical_cancelled":slot.cancelled,"result_published": !slot.cancelled,
                      "gpu_work_aborted":false,"physical_result":physical,"pending_snapshot":pending,
                      "commit_to_wait_seconds":ProcessInfo.processInfo.systemUptime-tick])
        try slot.recycle()
    }
    try rejection("count_zero") { try slot.fill(count: 0, run: run) }
    try rejection("count_over_capacity") { try slot.fill(count: capacity+1, run: run) }
    // Known-good real-device control precedes any cancellation fixture.
    for n in baseline { try execute("baseline-\(n)",n) }
    try slot.fill(count:65,run:run); try slot.cancel()
    let before = committed
    try rejection("submit_after_pre_cancel") { try slot.markSubmitted() }
    try require(committed == before, "pre-submit cancellation committed work")
    try slot.recycle()
    try execute("reuse", protocolData["reuse_size"] as! Int)
    try execute("logical-cancel-drain",protocolData["cancel_size"] as! Int,holdAndCancel:true)
    try execute("known-good-after-drain",protocolData["recovery_size"] as! Int)
    let expected = protocolData["expected_per_run"] as! [String:Int]
    try require(committed == expected["committed_command_buffers"] && callbacks == expected["completion_callbacks"] && guards.count == expected["guard_rejections"], "protocol totals mismatch")
    let report: [String: Any] = ["run":run,"stage":"device_run","outcome":"passed_bounded_runtime_fixture",
        "device":["name":device.name,"registry_id":String(device.registryID),"unified_memory":device.hasUnifiedMemory],
        "os":ProcessInfo.processInfo.operatingSystemVersionString,"shader_sha256":try digest(shaderURL),
        "protocol_sha256":try digest(protocolURL),"runner_sha256":try digest(runnerURL),
        "binary_sha256":try digest(URL(fileURLWithPath:CommandLine.arguments[0])),
        "runtime_pipeline_build_seconds":compileSeconds,"committed_command_buffers":committed,
        "completion_callbacks":callbacks,"cases":cases,"guard_rejections":guards,
        "physical_qualification":"not_run","sanitizer":"not_run","fault_injection":"not_run",
        "driver_firmware_build_identity":"not_exposed_by_selected_API",
        "scope":"trusted original toy; runtime-level shared-buffer ownership, logical cancel+drain; not authenticated coverage, physical DMA/driver/firmware/reset/thermal/portability qualification"]
    try JSONSerialization.data(withJSONObject: report, options:[.prettyPrinted,.sortedKeys]).write(to:outputURL,options:.withoutOverwriting)
    print("PASS run \(run): \(committed) Metal completions; \(guards.count) host guards; known-good after logical cancel+drain")
}
do { try main() }
catch { fputs("FAILED: \(error)\n", stderr); exit(1) }
