# become-human

尝试让AI以类人的方式行动。

原本叫aimemory。

[虽然不是同一个理想。](https://www.bilibili.com/video/BV1xH8oz8Eda)

## 安装使用

clone仓库或直接下载。

使用uv sync

设置.env

安装deno，linux或mac使用：

```
curl -fsSL https://deno.land/install.sh | sh
```

windows可以直接用仓库里的脚本，稍微改了一下，只会下载deno.exe到根目录，不会添加环境变量。这样似乎就够了。

uv run main.py/app.py即可或直接进入虚拟环境

网页服务（app.py）可以搭配 [become-human-app-assistant-ui](https://github.com/Bartzh/become-human-app-assistant-ui) 使用。

---

始终要做的：优化提示词，这个是真不会

已实现记忆管理（如果用RWKV会如何？）、打断（double-texting）、自我调用

TODO:
- 目前的回收机制有问题，太滞后
- agent自己的时间，自己的时区，自己的时间膨胀
- 使用模型计算初始stable_time

ROADMAP:
- 多模态
- 环境消息
- 角色设定（更新慢）：
    1. 自优化角色设定提示词
    2. 角色设定结构化
- 记住自身状态（更新快）：状态具体指哪些状态？
- 生理状态的处理机制（七情六欲？）
- MCP
- 多agent互动（supervisor）
- 终极目标：完全自思考，不停地自我调用，并根据环境行动