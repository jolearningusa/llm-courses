# 加速器协同设计：规格、代码与工作表

配套[第 40 讲](../../slides/zh/40_inference_accelerator_architectures.html) · [English](../accelerator_codesign_reference.md)。可选参考，核验于 2026-11-01。一手来源、排除的媒体参数与路线图取舍见[证据审查](../../references/accelerator_architectures_review.md)。未执行加速器或编译器基准。

**Google 行复核，2026-11-11：** [新增证据表](../../references/ironwood_spec_evidence_2026_11.md)更新该行的解读：单芯片分为两个独立内存空间，4 芯片 VM 在 JAX 中暴露 8 个设备；可用 HBM 低于标称值。7,380 GB/s 与正文约 7.37 TB/s 的差异原因未建立，不能把取整当作已验证解释。服务记录分别给出 2025-11-24 预览和 2026-03-31 GA；无本地 TPU / 框架资格验证。上面的日期保留为历史复核范围。

**真武行复核，2026-11-12 UTC：** [新增证据表](../../references/zhenwu_claim_evidence_2026_11.md)核对 5 月 / 9 月公告及 PAI 的 810E+SGLang 模板。保留 GB 与芯片间 GB/s，不推断 HBM 类型、可用容量、带宽方向或统一 3× 基准。V900 的 2027 Q1 仍为厂商预测。旧 keynote 链接两次超时，GPU 系列措辞未在本次重新确认。历史日期和旧解释保留；无硬件或软件兼容性实测。

**昇腾行复核，2026-11-12 UTC：** [工作模式证据表](../../references/ascend_mode_evidence_2026_11.md)区分共用 Scalar 的耦合模式和独立 Scalar 的分离模式。N 应由平台信息获取，不设固定比例；Atlas 200I/500 A2 的硬件能力与所选 Ascend C 文档的编程模式有区别。使用 CE/850 阅读范围，不宣称最新 SDK 支持；未查询设备、安装或执行。原复核日期保留为历史。

## 有来源的硬件边界

| 设备 | 文档确认的事实 | 边界 |
|---|---|---|
| Google TPU7x / Ironwood | 单芯片 BF16 2,307、FP8 4,614 TFLOP/s；HBM 192 GiB、约 7.4 TB/s；双向 ICI 1,200 GB/s。完整 Pod 9,216 芯片。 | [Google 表格](https://docs.cloud.google.com/tpu/docs/tpu7x)写 7,380 GB/s，正文约 7.37 TB/s，因此取整。峰值不是 token 吞吐。 |
| 华为昇腾 | AI Core 抽象含 Cube、Vector、Scalar 与数据移动资源。 | [华为文档](https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/tbeaicpudevg/atlasopdev_10_0008.html)；组织随产品变化，不推断统一核数或吞吐。 |
| 真武 810E | 文档列出 PPU-810E 部署选项。 | [阿里 PAI](https://www.alibabacloud.com/help/en/pai/use-cases/one-click-deploy-deepseek)；未经一手核实的媒体数值仍排除。 |
| 真武 M890 | 144 GB 内存、800 GB/s 芯片间带宽。 | [5 月官方公告](https://www.alibabacloud.com/en/press-room/alibaba-announces-comprehensive-full-stack-ai?_p_lc=1)，芯片间带宽不是 HBM 带宽。 |
| 真武 V900 | 216 GB 内存、1,200 GB/s 芯片间带宽；预计 2027 年第一季度量产与商用。 | [9 月官方公告](https://www.alibabacloud.com/en/press-room/alibaba-unveils-roadmap-on-full-stack-ai-strategy)，发布和预测不证明当前可用容量。 |

阿里将真武称为 GPU 系列，所选公开材料未证实其具体脉动或 SIMT PE 组织。厂商的 3 倍宣称不是统一披露的跨代基准，不应相乘为独立验证的端到端结论。SAIL 是已公布的软件栈；承诺集成前，仍需确认实际仓库、许可、芯片与 SDK/驱动组合及分析工具。证据审查记录已确认与未确认范围。

## 完整的单 tile 教学片段

课堂短代码用于展示抽象。下列扩展片段仍省略导入、装饰器、启动、索引、循环覆盖、边界或尾部处理及运行时设置，**不是独立程序**。假设 tile 完整且有效，输入输出 FP16、累加 FP32，BM/BN/BK 为受支持正值，Triton 指针块和 TileLang 张量区域已正确构建。

```python
# Triton：一个 K tile，三个指针块已正确构建。
acc = tl.zeros((BM, BN), tl.float32)
a = tl.load(a_ptrs)
b = tl.load(b_ptrs)
acc = tl.dot(a, b, acc)
tl.store(c_ptrs, acc.to(tl.float16))
```

```python
# TileLang：一个 K tile，假设全局张量区域已正确构建。
f16, f32 = "float16", "float32"
shape_a, shape_b, shape_c = (BM, BK), (BK, BN), (BM, BN)
As = T.alloc_shared(shape_a, f16)
Bs = T.alloc_shared(shape_b, f16)
Cf = T.alloc_fragment(shape_c, f32)
T.clear(Cf)
T.copy(A_tile, As)
T.copy(B_tile, Bs)
T.gemm(As, Bs, Cf)
T.copy(Cf, C_tile)  # C_tile 是 FP16 输出区域。
```

缺失的程序结构参见 [Triton 完整 GEMM 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)与[固定修订 TileLang 示例](https://github.com/tile-ai/tilelang/blob/e8225d7f5e8457c1360386e0603bc8196dcff095/examples/gemm/example_gemm.py)。[固定修订语言指南](https://github.com/tile-ai/tilelang/blob/e8225d7f5e8457c1360386e0603bc8196dcff095/docs/programming_guides/language_basics.md)说明 shared/fragment 存储域及手动异步拷贝的等待要求。自动布局与同步 pass 不能证明任意程序正确，也不证明支持所有被提及芯片。此次教学修改未编译这些片段。

## 分析假设与实验工作表

阵列预测题刻意固定 256 行 tile，其中一行有效，得到 1/256 行占用率，不是整体芯片利用率实测。输出驻留调度的 `t=k+i+j` 假设零起始索引、错位注入与每跳一拍。图只突出教学节拍 2 的 k=0 波前，省略其他 k 数据流。理想 GEMM 流量假设输入各读一次、输出写一次且无溢出；`sA/sB/sC` 是每存储元素字节数。

固定模型或算子、形状、精度、质量容差、延迟目标和总资源，分别保留 prefill 与 decode 案例。每次改变一种 tile、布局、流水或量化路径，检查生成代码和资源，再先验输出后计时。分别记录编译启动、预热后的稳态、主机搬运和通信。仅有支持证据时才把收益归因于复用、占用率或开销减少。加入无收益或更差的负载；没有合适设备时明确报告目标不受支持。

## 固定版本的完整内核导读 — 2026-11-09

[TileLang GEMM 源码与硬件映射补充](../../references/tilelang_gemm_walkthrough_2026_11.md)跟踪提交 `72d986e1c45ba8e81f7611eb6835c9bece097d60` 的启动网格、shared/fragment 存储、K 流水线、输出转换，以及仍需核对的边界与同步。第 40 讲第 19/21 页配有对应图。上面的单块片段仍是概念教学；补充材料提供作者完整源码的固定阅读版本，不代表课程已编译内核。工作表列出完整块、M/N/K 不规则尾部与小 M 对照。本轮只执行整数覆盖推导，数值与设备验证以及 FA4/CuTeDSL 导读仍待完成。
