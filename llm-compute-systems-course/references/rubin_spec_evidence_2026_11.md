# Rubin specification evidence: units, conditions and status — 2026-11-10

CMP-4's **selected Rubin rows**, based on main `8089d06`. [Deck 24](../slides/24_rubin_vendor_stacks.html) slides 6/7 and their [Chinese twin](../slides/zh/24_rubin_vendor_stacks.html) now visualize source boundaries. The [machine-readable ledger](rubin_spec_evidence_2026_11.json) records selected values, exact URLs, dates, PDF hash and arithmetic. The [original case study](../notes/vera_rubin.md) retains its 2026-10-27 review as history; this supplement does not re-certify all architecture/library claims.

## Read the column before the number

The [current NVL72 product table](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) has columns **NVL72 / Superchip / GPU**. The vendor-linked [datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/56p68o47y6f1yucpubl0ump1c0aif422.pdf), printed page 8, instead has **NVL72 / Superchip / NVL4**; its third column is not a GPU. Printed page 9 supplies the sparse/dense/networking footnotes; page 3 contains the workload-comparison conditions. Those complete pages were rendered and visually inspected, including headings and footnotes.

| Resource | One GPU | Superchip | NVL72 |
|---|---:|---:|---:|
| GPUs / Vera CPUs | 1 / — | 2 / 1 | 72 / 36 |
| HBM capacity | 288 GB | 576 GB | 20.7 TB |
| HBM bandwidth | 19.2 TB/s | 38.5 TB/s | 1,400 TB/s |
| NVLink bandwidth | 3 TB/s | 6 TB/s | 216 TB/s |

These are published specifications, not local measurements. Rack totals do not make all HBM equally local, one contiguous allocation, one kernel's memory rate or a network bisection rate. For illustration, `72×288 GB=20.736 TB` using decimal unit conversion, while the table publishes 20.7 TB; `72×19.2=1382.4 TB/s` versus its published 1,400 TB/s. Finite reported precision may matter, but this review does not establish a rounding or operating-point explanation. Arithmetic supplies no achieved throughput. The 100 MW/40K-GPU factory table is another system boundary.

## Keep the source discrepancy open

The product table still lists per-GPU HBM/NVLink rates **19.2 / 3 TB/s**. The [July 21 architecture article](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/), body sections on architecture and memory, gives **up to 22 / 3.6 TB/s**. This review does not establish whether SKU, clocks, power, implementation or publication revision explains the difference. Do not select the larger pair, average them, or claim a documented upgrade. The earlier two-source observation remains valid as a source discrepancy.

Do not apply the table's footnote 4, attached to the networking row, automatically to every bandwidth. The selected NVLink row does not itself resolve directionality. State link type, endpoint/aggregate scope, payload versus signaling and direction convention before using any rate in an estimate. HBM, C2C, NVLink and scale-out bandwidths are not additive token throughput.

## Precision peaks and workload comparisons need different contracts

The product table labels per-GPU **50 PFLOPS NVFP4 inference as sparse** and **4 PFLOPS FP16/BF16 as dense**. Different precision/sparsity contracts cannot be divided into an application speedup. Model accuracy, quantization overhead, shapes and achieved utilization still need execution evidence.

The datasheet's page 3 comparison is a vendor **up-to 10× tokens/s/MW** claim for Kimi K2 Thinking at 32K/8K input/output lengths against GB200 NVL72, subject to change. Its **one-quarter GPU-count** training comparison is separately projected for a 10T MoE / 100T-token workload completed in one month. Neither is a generic 10× speedup, an inference-only platform claim or a course benchmark. Comparisons involving additional LPX racks must retain that broader system boundary.

## Availability is not laboratory qualification

On this check, the product page labels availability as **Available Now**; the [May 31, 2026 newsroom release](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory) reports a production ramp. These are dated vendor statements, not verification of a particular region, order, installed rack or software combination. No local Rubin device, driver/toolkit/library dispatch, model quality, latency or power was tested.

The 11-page PDF downloaded from the vendor link is 1,172,802 bytes with SHA-256 `6b04af369412dc707d3be14a8fed5d981b7987a411f542245f9094321e452f3d`. Its metadata timestamps do not establish a publication date or hardware availability. The upstream PDF and its figures are linked, not redistributed; new slide SVGs are original teaching schematics.

## L15 acceptance and remaining audit scope

Use the [vendor dossier](../labs/kit/vendor_dossier_template.md): preserve source date, table heading, units, precision/sparsity, boundary, workload/baseline and evidence class. Reject a dossier that turns the PDF's NVL4 column into one GPU, combines two published rate pairs into an invented SKU, treats sparse NVFP4 as dense BF16 performance, or reports vendor availability as an executed course result.

This implements the bounded Rubin specification/claim review only. Google/Ironwood, Ascend/T-Head and AMD device/library rows remain separate CMP-4 work; no new assertion about those vendors is made. Real topology, memory availability, supported software and workload performance remain unverified. No lab Python, old numerical result table or tracked dependency pin changes in this update.
