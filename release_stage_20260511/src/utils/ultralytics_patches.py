"""Ultralytics 兼容补丁。

本项目的注意力对比实验使用 Ultralytics 已内置的 CBAM、ChannelAttention
和 SpatialAttention 模块。部分版本虽然在 `ultralytics.nn.modules` 中导出了
这些模块，但 `parse_model` 使用的 `ultralytics.nn.tasks` 全局命名表中没有注册，
会导致自定义 yaml 中写 `CBAM` 时出现 `KeyError: 'CBAM'`。

这里在项目入口处做显式注册，不修改第三方库源码，便于本地和 Slurm 服务器复现。
"""


def register_attention_modules() -> None:
    """把 Ultralytics 内置注意力模块注册到模型解析器命名空间。"""

    try:
        from ultralytics.nn import modules, tasks
    except ImportError:
        return

    for name in ("CBAM", "ChannelAttention", "SpatialAttention"):
        if hasattr(modules, name):
            setattr(tasks, name, getattr(modules, name))

