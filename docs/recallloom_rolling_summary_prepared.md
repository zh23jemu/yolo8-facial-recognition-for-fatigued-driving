<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [codex] | 2026-05-12 -->

<!-- section: current_state -->
# 当前状态

- 项目主体已经具备训练、推理、桌面演示、论文写作和交付打包能力。
- 论文目录已重组为“相关技术理论基础、数据集构建与预处理、模型/算法设计与改进、实验设计与结果分析、系统设计与实现”的结构。
- 已生成两个主要用户交付包：一个只含程序与模型，另一个包含程序、模型与 `data.zip` 数据集。
- 当前最新的带数据集交付包为 `fatigued_driving_program_models_dataset_20260512.zip`。
- 客户电脑上曾出现 `Could not find the Qt platform plugin "windows"` 报错，现已在 `src/app/main_window.py` 中加入 Qt 插件路径自动配置逻辑。
- `RecallLoom` 已于 2026-05-12 初始化并完成首轮项目记忆写入，当前项目已具备可恢复的连续性上下文。

<!-- section: active_judgments -->
# 当前判断

- 该项目属于“论文 + 算法实验 + 桌面演示 + 最终交付”混合型项目，后续记录需要同时覆盖代码、文档和打包状态。
- 对外发包时，优先提供已经修复 Qt 插件路径问题的新版压缩包，而不是旧版源码包。
- 客户电脑适配的长期最稳方案不是继续依赖源码 + `.venv`，而是后续考虑制作独立可执行发布版。

<!-- section: risks_open_questions -->
# 风险与未决问题

- 目前虽然已经补了 Qt 插件路径自动配置，但尚未在真正的客户电脑环境上完成回归验证。
- 交付方式仍以源码包为主，客户环境若缺少正确依赖或本地 Python 环境异常，仍可能出现新的运行问题。
- 项目已有多个历史压缩包，后续继续交付时要明确区分最新推荐包，避免用户拿到旧版本。

<!-- section: next_step -->
# 下一步

- 优先让客户使用 `fatigued_driving_program_models_dataset_20260512.zip` 再次验证桌面程序启动情况。
- 如果客户环境仍不稳定，下一步应改做 Windows 独立发布版，降低对本地 Python/Qt 环境的依赖。
- 后续若继续推进论文或交付，应同步更新 `RecallLoom` 记录，确保最新包名、关键修复和未决风险保持一致。

<!-- section: recent_pivots -->
# 近期判断反转

- 早期交付默认以源码包为主；在客户电脑出现 Qt 平台插件错误后，当前判断转向“交付稳定性优先”，并优先推动更强的环境自适应或独立发布方案。
