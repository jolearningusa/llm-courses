# 推理引擎：参考矩阵与实验工作表

配套[第 35 讲](../../slides/zh/35_inference_engines_compared.html) · [English](../inference_engines_reference.md)。2026-11-01 按项目来源核验。这里是可选参考，不是速度排名或已执行基准。省略版本排行榜与热度数，实验应记录实际测试修订。

## 候选路径

| 项目 | 有用的比较问题 | 原始来源 |
|---|---|---|
| llama.cpp / llama-server | ggml 后端、卸载、槽位和缓存设置如何影响这个模型？ | [仓库](https://github.com/ggml-org/llama.cpp)、[服务参考](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) |
| Ollama | 模型管理与 API 层为本次模型选了什么执行后端和默认值？ | [仓库](https://github.com/ollama/ollama)、[引擎说明](https://ollama.com/blog/multimodal-models) |
| MLX / MLX-LM | 表示、驻留与 Apple silicon 路径如何影响负载？ | [MLX-LM](https://github.com/ml-explore/mlx-lm)、[服务边界](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) |
| vLLM | 实际启用了哪些调度、分页 KV 与前缀复用路径？ | [仓库](https://github.com/vllm-project/vllm)、[GGUF 状态](https://docs.vllm.ai/en/stable/features/quantization/gguf/) |
| SGLang | 前缀感知复用是否改善这种到达与历史模式？ | [仓库](https://github.com/sgl-project/sglang)、[文档](https://docs.sglang.io/) |
| TensorRT-LLM | 哪组受支持 NVIDIA 精度、kernel 与运行时路径适合任务？ | [仓库](https://github.com/NVIDIA/TensorRT-LLM)、[文档](https://nvidia.github.io/TensorRT-LLM/) |

这些是候选问题，不是硬边界。API 兼容需按端点和功能检查；共同的聊天请求不保证采样、工具、调度或数值行为相同。

## 值得记录的配置字段

| 关注点 | llama-server 中可检查的示例 | 其他引擎中的对应问题 |
|---|---|---|
| 放置 | `--gpu-layers`、`--split-mode`、`--tensor-split` | 设备放置、分片与搬运边界 |
| 并发 | `--parallel`、合批设置 | 准入、活跃序列与 token 预算 |
| KV 表示 | `--cache-type-k`、`--cache-type-v` | 缓存 dtype、布局与质量 |
| 前缀复用 | 请求缓存控制与复用策略 | 缓存键、因果有效性与命中率 |
| 缓存层级 | 版本支持时的 `--cache-ram` 等配置 | 保留字节、搬运与重算成本 |
| 推测 | 推测类型与草稿模型 | 接受率、验证工作与端到端收益 |

以安装版本的帮助为准，记录实际值而非背默认值。核验的服务参考列出 `none`、`layer`、`row` 与**实验性 `tensor`** 分割，因此“llama.cpp 没有张量并行路径”过于绝对。反过来，列出 tensor split 也不能证明生产扩展或 prefill/decode 拆分行为与其他引擎相同。

[GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)定义可扩展元数据与张量格式，支持内存映射及分片文件；它不保证加载器实现每种模型和算子。权重量化、KV 量化与校准是独立选择。Q4_K_M 等名称描述配方，标称位宽不能使两个表示数值等价。

MLX 统一内存改变放置与显式拷贝选择，不消除带宽、同步或内部搬运。MLX-LM 服务文档说明它不推荐用于生产。vLLM 的 GGUF 文档把该路径标为实验性、优化不足，并说明插件要求；应核对版本边界，不把“可加载”当作相同优化路径。也不要从一个版本推断 Ollama 后端永不变化，应记录被测模型与实际后端。

## 实验工作表

1. 固定模型修订、tokenizer/chat template、任务与质量标准、采样、输入输出长度及到达过程。
2. 记录设备、驱动、引擎与后端修订、权重和量化校准、KV dtype、放置与非默认选项。
3. 声明缓存重置，比较冷前缀与共享前缀。固定资源下做小范围并发扫描，例如仅在支持时使用 1、8、32；不要把缓存预热偷换为引擎加速。
4. 报告 TTFT、逐 token 延迟、有效完成 token、尾延迟与 SLO 达标、拒绝取消失败请求及峰值内存。保留原始请求记录，分开记录启动和编译。
5. 用第 15–17 讲的合批、缓存有效性、kernel、表示与放置解释结果，列明所有未匹配变量。

无法匹配表示或质量时，标为**配置比较**，而非孤立引擎效应。范围明确时，这仍可支持部署决策。现有实验入口见课件中的 L7、L8 链接；本讲义不宣称新安装了引擎或取得 GPU 结果。

[RadixAttention 机制与历史评估](https://lmsys.org/blog/2024-01-17-sglang/)：阅读机制，不把历史加速比当作当前排名。
