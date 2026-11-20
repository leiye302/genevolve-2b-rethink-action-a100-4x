# GenEvolve 2B RETHINK：统一 action 协议（4×A100 80GB）

这是独立的**动作协议对照仓库**：从 [原 RETHINK 仓库 b18f659](https://github.com/leiye302/genevolve-2b-rethink-a100-4x/tree/b18f659e563d36fc161357da5ddfd65f4266af8f) 出发，只将双分支 `<tool_call>` / `<answer>` 改成 ALFWorld 风格的 **`<action>命令名 原参数</action>`**。原仓库及其 type-BCE 实验不受影响。

保留同一份 **Qwen3-VL-2B GenEvolve SFT-550** 起点、原 Route-SFT 配方，以及二阶段 **GenEvolve GRPO + SDL + Actor-routed Teacher-only hint**。不重训 SFT-550，不引入 type BCE、reasoning KL 或额外辅助阶段。不把 GenEvolve SDL 改成 SDAR 的蒸馏或门控。

**训练轮次更新：**本仓库与原生 type-BCE 版均改为 **1 epoch / 297 updates**。除这项明确的训练时长调整外，协议对照的其他设置不变。

## 1. 新动作协议

每回合只有一个 action，下面四行是四种选择，不是一次输出四个动作：

```text
<action>search {"queries":["missing factual evidence"]}</action>
<action>image_search {"query":"visual reference"}</action>
<action>query_knowledge {"skill_name":"spatial_layout"}</action>
<action>generate {"gen_prompt":"...","reference_images":[{"img_id":"IMG_001","note":"..."}]}</action>
```

```text
Good: <think>原 reasoning</think>\n<action>命令名 原采样参数</action>
Bad:  <think>原 reasoning</think>\n<RETHINK><action>命令名 已审核的纠正参数</action>
```

`generate` 对应原 `<answer>`：提交原来的 prompt-reference program 并结束轨迹，之后仍由原渲染器绘图；不是新增一次生成工具调用，也不是丢掉原参数的 `finish`。三种工具的名称、参数、默认 top-k 和反馈完全保留。Actor 仍只看文字；图片下载和参考图选择照旧，不额外把图片送入 Actor。

提示词、离线历史、监督后缀、在线解析、蒸馏 action span 和评测使用同一协议。原最后一轮的“直接交最终结果、不要写 think”例外也保留，仅把输出格式改为 `generate`；没有新加强制前缀或约束解码。详见 [协议与改动边界](docs/PROTOCOL.md)。

## 2. 数据已经随 Release 发布

[下载完整 action 版 Route-SFT 数据包](https://github.com/leiye302/genevolve-2b-rethink-action-a100-4x/releases/tag/route-sft-command-first-v1)。39.4 MB 压缩包，内含 `manifest.json`、`train.jsonl`、`batches.jsonl`；不需要私人文件、原始轨迹或本机目录。默认命令会自动下载并校验。

| 数据 | 数量 |
| --- | ---: |
| Good | 1,782 |
| Bad | 594（394 自然 + 200 同状态合成） |
| 总计 | 2,376 |
| 固定批次 | 297，每批 6 Good + 2 Bad |
| 训练轮次 / updates | 1 / 297 |
| 最终 `generate` 目标 | 621（492 Good + 129 Bad） |

没有换样本、标签或纠正内容。转换核对了全部 2,376 个目标和 6,091 段历史 assistant 响应的执行含义；当前 reasoning 文本和 token 保持原样，批次文件逐字节不变。**动作格式改变，所以受监督 action token 必然重新编码**，不能称整个 token 包与旧版相同。最长 11,078 tokens，上限仍为 12,288，超长报错、不截断。原数据中一个历史格式失败仍保留为失败，没有伪造成功反馈。[数据说明](DATASET.md) · [机器可读核对结果](data/protocol_audit.json)。

## 3. 一阶段损失不变，训练一轮

令 $s_i$ 为历史与 reasoning 结束后的路由位置，$y_i$ 为 Bad 标签，$A_i$ 为完整监督后缀（含 Bad marker 及 EOS）：

$$
\mathcal L=\frac1B\sum_{i=1}^B\left[
-\frac1{|A_i|}\sum_{t\in A_i}\log p_\theta(a_{i,t}\mid H_i,R_i,a_{i,<t})
+0.1\,\operatorname{BCE}\!\left(p_\theta(\texttt{<RETHINK>}\mid s_i),y_i\right)\right].
$$

- mask 到 `</think>\n`（含分隔符），只监督后缀和 EOS。marker BCE 取首个目标 token **之前**的预测，使用全词表概率，不用原始 marker logit 直接 sigmoid。
- CE 先对每个样本后缀取均值，再跨样本取均值；不增加命令类型均衡、重采样或长度归一化新方案。
- 全局 batch 8 = 四卡 × micro 1 × 累积 2。AdamW，LR `2e-6`，betas `(0.9,0.95)`，weight decay `0.01`，warmup `10%`，cosine，clip `1.0`，seed `0`。
- 语言参数全参训练，视觉塔冻结；无 reasoning CE、无 KL、无 LoRA。只注册一个 `<RETHINK>`；`<action>` 和命令使用原 tokenizer。
- 单轮结束保存可续训状态和 HF 导出（297 步）；cosine 总步数改为 297，warmup 仍按 10% 取整。本包没有额外 SFT val split；给定前缀 marker 概率不是自由生成触发率。

## 4. 二阶段与评测不变的部分

| 项目 | 原设置 |
| --- | --- |
| RL 数据 | 固定版本，2,575 题；seed 42 |
| RL 每步 | 8 题 × 每题 6 条轨迹；四卡各 2 个完整题组 |
| RL 采样 / 预算 | temperature 0.7，top-p 0.95；11 回合，每回合 4,096 新 token，总 30,000；初始输入 6,144、上下文 65,536 |
| GRPO | 原组内标准化，0.20/0.28 裁剪，奖励 KL 0.001，程序奖励 0.5，LR 1e-6，cosine |
| SDL | 原高低奖励轨迹配对、经验提取与检索；action span 内 top-10%，系数 2.0 |
| 局部 hint | Actor marker 单独决定请求；更新前同一 Actor 的无梯度 Teacher 评分；Student 永远看不到 hint |
| hint 失败 | 有限重试后回退原经验 Teacher；无经验则不做该项蒸馏，保留 GRPO |
| 保存 | 每 10 步和最后一步；最近两份可续训断点 + 最终 HF 导出 |
| 渲染 | 固定 Qwen-Image-Edit-2511，40 步、CFG 4、seed 0、最多两张参考图 |
| 评分 | 基线 `qwen3.7-flash-2026-07-15` 奖励 / KScore Judge；不作为 hint 模型 |
| 正式终评 | GenEvolve-Bench 同序 594 题、原评分与原 prompt fallback；temperature 0.6 / top-p 0.9；seed=42+题序 |

完整轨迹照常 GRPO。未触发步骤仍用原检索经验；触发步骤给 Teacher 加局部 hint，允许 hint 在空经验库时单独生效。API 不判断 Good/Bad、不决定路由、不生成替代执行动作、不覆盖经验库。marker 位置正确率仅用于统计，不是第二个门控。

默认二阶段 `ceil(2575/8)=322` updates。需要 40/50 步等实验预算时显式传 `--steps N`，配对组使用相同计划步数；cosine 以计划长度计算。旧 native 协议 / type-BCE 的 Route-SFT 或二阶段断点不能直接接这版，会被配置及数据指纹校验拒绝。

## 5. 从干净机器运行

需要 Linux x86-64、Python 3.11、CUDA 12.8 toolkit 和 C/C++ 构建工具。沿用四张 A100 80GB、FSDP + CPU offload。每阶段首次启动检查至少 100 GiB 可用盘；同盘已有断点、渲染缓存另计。真实 GPU 峰值和吞吐未在这次发布中实测。

运行者提供与对照组相同的 **2B SFT checkpoint-550 HF 导出**（权重、tokenizer、processor、`trainer_state.json`）。它是训练起点，不包含在 39.4 MB 数据包里；可由[固定版本的原 2B 复现仓库](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/57263f0583c633c3d515913b9935c3850cdf798e)产出。无需改它的 550 步训练，也不接原始基模或 8B 权重。

```bash
git clone https://github.com/leiye302/genevolve-2b-rethink-action-a100-4x.git
cd genevolve-2b-rethink-action-a100-4x
bash scripts/setup.sh
source .venv/bin/activate
export WORK=/path/to/new-independent-run
export PARENT=/path/to/genevolve-2b-sft/checkpoint-550
python run.py --work "$WORK" prepare-route
python run.py --work "$WORK" check-route --parent "$PARENT"
# 只训练一阶段：不需要任何付费 API
python run.py --work "$WORK" route-sft --parent "$PARENT"
```

`check-route` 用实际起点的 tokenizer 逐条检查历史、mask、解码和边界，不一致直接停。原离线数据由 8B 采集，训练起点为 2B；两个概念不可混用。完整来源指纹已记录，不依赖目录名字判断模型。

二阶段及评测需要工具 / 渲染 / Judge 服务；方法训练另需便宜的 `glm-5.3-flash` 局部 hint API。参见 [环境变量模板](configs/services.env.example)。请确认服务商允许接收所用训练文本，凭据仅由运行者注入：

```bash
export QWEN_SERVICE_URL=http://render-host:8001
export GENEVOLVE_HINT_API_URL=https://opencode.ai/inference/go/openai/v1/chat/completions
read -rsp 'Serper key: ' SERPER_API_KEY; printf '\n'
read -rsp 'Judge key: ' JUDGE_API_KEY; printf '\n'
read -rsp 'Hint key: ' GENEVOLVE_HINT_API_KEY; printf '\n'
export SERPER_API_KEY JUDGE_API_KEY GENEVOLVE_HINT_API_KEY
python run.py --work "$WORK" prepare
python run.py --work "$WORK" rethink --sft-checkpoint "$WORK/route_sft/hf-step-00297" --steps 50
python run.py --work "$WORK" eval --split bench
```

也可以用 `python run.py --work "$WORK" all --parent "$PARENT" --steps 50 --eval` 串联两阶段和终评，已完成的一阶段会校验后复用。经验检索模型自动按固定版本下载；渲染服务可另机执行 `bash scripts/setup.sh --renderer` 和 `bash scripts/serve_renderer.sh`。同命令、同 WORK 重启可从完整断点续跑，不依赖发布者电脑在线。

**比较口径：** 本仓库所有生成入口都用新协议，包括可选 `grpo` 和 `eval --policy`。未经协议适配的 SFT-550 在新协议下的结果不是“原生基线”。原生基线请使用[原 GRPO 仓库](https://github.com/leiye302/genevolve-2b-grpo-a100-4x/tree/2f197591100d8b0bf61e65be9748a0752f91ae30)或 [GenEvolve 复现仓库](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/57263f0583c633c3d515913b9935c3850cdf798e)，并报告协议差别；要隔离协议影响，再做同协议消融。

## 6. 检查与来源

```bash
bash scripts/setup.sh --sources-only
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-smoke.txt
python -m unittest discover -s tests -v
```

CPU 测试覆盖新旧执行循环的同义对照、动作/历史/最后一轮、参考图、marker 解码与监督位置、后缀 CE/BCE 梯度、Teacher-only hint、原 SDL、断点及下载校验。测试中的工具、渲染器和 API 使用替身，不据此宣称 GPU 训练完成或性能提高。

作者 GenEvolve 源码由安装脚本下载到仓库内 `.deps` 并固定提交，见 [sources.json](sources.json)。新旧核心实现核对见 [method_diff.json](data/method_diff.json)。代码遵循来源的许可证，见 [NOTICE.md](NOTICE.md)。
