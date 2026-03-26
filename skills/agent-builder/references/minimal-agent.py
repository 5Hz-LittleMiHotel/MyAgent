#!/usr/bin/env python3
"""
最简 Agent 模板 - 复制后按需定制。

这是能正常运行的最精简 Agent（约 80 行）。
包含所有必要组成部分：3 个工具 + 主循环。

使用方法：
    1. 设置环境变量 ANTHROPIC_API_KEY
    2. python minimal-agent.py
    3. 输入指令，输入 'q' 退出
"""

from anthropic import Anthropic   # 导入 Anthropic 官方客户端库
from pathlib import Path          # 导入路径操作模块，用于跨平台文件路径处理
import subprocess                 # 导入子进程模块，用于执行 shell 命令
import os                         # 导入 os 模块，用于读取环境变量

# ── 基础配置 ──────────────────────────────────────────────────────────────────

# 使用环境变量中的 API Key 初始化 Anthropic 客户端
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# 模型名称，优先读取环境变量，未设置则使用默认值
MODEL = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")

# 工作目录，默认为当前目录（脚本运行所在目录）
WORKDIR = Path.cwd()

# ── 系统提示词 ─────────────────────────────────────────────────────────────────

# 告知模型其角色和行为规范；f-string 将工作目录动态注入提示词
SYSTEM = f"""You are a coding agent at {WORKDIR}.

Rules:
- Use tools to complete tasks
- Prefer action over explanation
- Summarize what you did when done"""

# ── 工具定义列表 ───────────────────────────────────────────────────────────────

# TOOLS 是传给 API 的工具描述，每个元素对应一个可调用工具
TOOLS = [
    {
        "name": "bash",                          # 工具名称，模型调用时使用此名字
        "description": "Run shell command",      # 工具用途描述，帮助模型决定何时调用
        "input_schema": {                        # 定义该工具接收的参数结构（JSON Schema）
            "type": "object",
            "properties": {"command": {"type": "string"}},  # 唯一参数：shell 命令字符串
            "required": ["command"]              # command 为必填参数
        }
    },
    {
        "name": "read_file",                     # 读取文件工具
        "description": "Read file contents",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},     # 参数：文件路径字符串
            "required": ["path"]
        }
    },
    {
        "name": "write_file",                    # 写入文件工具
        "description": "Write content to file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},      # 参数1：目标文件路径
                "content": {"type": "string"}    # 参数2：要写入的文本内容
            },
            "required": ["path", "content"]      # 两个参数均为必填
        }
    },
]

# ── 工具执行函数 ───────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    """根据工具名称执行对应操作，返回字符串结果。"""

    if name == "bash":
        try:
            r = subprocess.run(
                args["command"],        # 要执行的 shell 命令
                shell=True,             # 通过 shell 解释器执行，支持管道、通配符等
                cwd=WORKDIR,            # 指定命令的工作目录
                capture_output=True,    # 捕获 stdout 和 stderr，不直接打印到终端
                text=True,              # 将输出以字符串形式返回（而非 bytes）
                timeout=60              # 超时限制：60 秒，防止命令挂起
            )
            # 合并标准输出与错误输出；若均为空则返回占位字符串
            return (r.stdout + r.stderr).strip() or "(empty)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout"     # 超时时返回错误提示

    if name == "read_file":
        try:
            # 拼接工作目录与相对路径，读取文件文本内容，最多返回 50000 个字符
            return (WORKDIR / args["path"]).read_text()[:50000]
        except Exception as e:
            return f"Error: {e}"        # 文件不存在或权限不足时返回错误信息

    if name == "write_file":
        try:
            p = WORKDIR / args["path"]           # 构造完整目标路径
            p.parent.mkdir(parents=True, exist_ok=True)  # 递归创建父目录，已存在则忽略
            p.write_text(args["content"])        # 将内容写入文件（覆盖已有内容）
            return f"Wrote {len(args['content'])} bytes to {args['path']}"  # 返回写入字节数
        except Exception as e:
            return f"Error: {e}"

    # 若工具名不匹配以上任何分支，返回未知工具提示
    return f"Unknown tool: {name}"

# ── Agent 主循环函数 ────────────────────────────────────────────────────────────

def agent(prompt: str, history: list = None) -> str:
    """执行 Agent 对话循环，直到模型不再调用工具时返回最终文本回复。"""

    if history is None:
        history = []                    # 首次调用时初始化空对话历史

    # 将用户输入追加到对话历史
    history.append({"role": "user", "content": prompt})

    while True:
        # 调用 Claude API，传入模型名、系统提示、历史消息、工具定义和最大 token 数
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=history,
            tools=TOOLS,
            max_tokens=8000,
        )

        # 将本轮模型回复（含可能的工具调用块）追加到历史，保持对话上下文连贯
        history.append({"role": "assistant", "content": response.content})

        # 若停止原因不是工具调用，说明模型已完成任务，提取纯文本并返回
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        # ── 执行本轮所有工具调用 ──
        results = []
        for block in response.content:
            if block.type == "tool_use":                    # 只处理工具调用类型的块
                print(f"> {block.name}: {block.input}")     # 打印工具名和参数，便于调试
                output = execute_tool(block.name, block.input)   # 实际执行工具
                print(f"  {output[:100]}...")               # 打印输出的前 100 个字符预览
                results.append({
                    "type": "tool_result",       # 固定类型标识，API 要求
                    "tool_use_id": block.id,     # 与对应工具调用的 id 匹配，API 要求
                    "content": output            # 工具执行结果字符串
                })

        # 将所有工具结果作为 user 角色消息追加到历史，供下一轮模型读取
        history.append({"role": "user", "content": results})

# ── 入口：交互式命令行界面 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Minimal Agent - {WORKDIR}")   # 启动时打印工作目录
    print("Type 'q' to quit.\n")

    history = []                          # 跨轮次共享的对话历史，保持上下文记忆
    while True:
        try:
            query = input(">> ").strip()  # 读取用户输入并去除首尾空白
        except (EOFError, KeyboardInterrupt):
            break                         # Ctrl+D 或 Ctrl+C 时优雅退出
        if query in ("q", "quit", "exit", ""):
            break                         # 输入退出指令或空行时结束循环
        print(agent(query, history))      # 调用 agent，打印最终回复
        print()                           # 输出空行，提升可读性