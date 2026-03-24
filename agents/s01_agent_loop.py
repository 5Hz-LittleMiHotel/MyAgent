#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""


import os  # 操作系统相关（环境变量、路径等）
import subprocess  # 用来执行 shell 命令

from anthropic import Anthropic  # Anthropic 官方 SDK
from dotenv import load_dotenv  # 用于加载 .env 环境变量文件

# 加载当前目录下的 .env 文件（覆盖已有环境变量）
load_dotenv(override=True)

# 如果设置了自定义 API 地址（例如代理/本地服务）
if os.getenv("ANTHROPIC_BASE_URL"):
    # 则移除默认 token（避免冲突）
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 创建 Anthropic 客户端（可连接官方或自定义 endpoint）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# 从环境变量中读取模型 ID（比如 claude-3）
MODEL = os.environ["MODEL_ID"]

# 系统提示词（告诉模型它是谁、该做什么）
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# 定义工具（这里我们只给模型一个工具：bash）
TOOLS = [{
    "name": "bash",  # 工具名字
    "description": "Run a shell command.",  # 工具描述
    "input_schema": {  # 输入参数格式（JSON Schema）
        "type": "object",
        "properties": {
            "command": {"type": "string"}  # 要执行的命令
        },
        "required": ["command"],  # 必填字段
    },
}]


def run_bash(command: str) -> str:
    """
    执行 bash 命令的函数
    带有基础安全保护（防止误删系统等）
    """

    # 定义危险命令（简单黑名单）
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    # 如果命令中包含危险内容，直接拒绝执行
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # 执行命令
        r = subprocess.run(
            command,
            shell=True,  # 通过 shell 执行
            cwd=os.getcwd(),  # 当前工作目录
            capture_output=True,  # 捕获 stdout/stderr
            text=True,  # 返回字符串而不是字节
            timeout=120  # 最长执行 120 秒
        )

        # 合并标准输出和错误输出
        out = (r.stdout + r.stderr).strip()

        # 限制输出长度（防止爆内存）
        return out[:50000] if out else "(no output)"

    except subprocess.TimeoutExpired:
        # 超时保护
        return "Error: Timeout (120s)"


# -- 核心 Agent 循环 --
def agent_loop(messages: list):
    """
    这是整个 Agent 的核心循环：

    不断：
    1. 调用 LLM
    2. 如果 LLM 想用工具 -> 执行
    3. 把结果喂回去
    直到 LLM 停止
    """
    # messages = [{"role": "user", "content": query}]
    while True:
        # 调用 LLM（带上下文 + 工具定义）
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        # 把模型的回复加入对话历史
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        # 如果模型没有调用工具 -> 结束循环
        if response.stop_reason != "tool_use":
            return

        # 否则：执行工具调用
        results = []

        # response.content 是一个 block 列表
        for block in response.content:

            # 如果这个 block 是工具调用
            if block.type == "tool_use":

                # 打印命令（黄色）
                print(f"\033[33m$ {block.input['command']}\033[0m")

                # 执行 bash
                output = run_bash(block.input["command"])

                # 打印部分输出（防止太长）
                print(output[:200])

                # 收集结果（返回给模型）
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # 对应 tool_use 的 ID
                    "content": output
                })

        # 把工具执行结果作为“用户消息”喂回模型
        messages.append({
            "role": "user",
            "content": results
        })


# 程序入口
if __name__ == "__main__":

    history = []  # 对话历史（messages）

    while True:
        try:
            # 读取用户输入（带颜色提示）
            query = input("\033[36ms01 >> \033[0m")

        except (EOFError, KeyboardInterrupt):
            # Ctrl+D 或 Ctrl+C 退出
            break

        # 输入 q / exit / 空字符串 -> 退出
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 把用户输入加入历史
        history.append({
            "role": "user",
            "content": query
        })

        # 运行 agent 循环
        agent_loop(history)

        # 取最后一条消息（模型输出）
        response_content = history[-1]["content"]

        # 如果是结构化 block（列表）
        if isinstance(response_content, list):
            for block in response_content:
                # 有 text 属性就打印
                if hasattr(block, "text"):
                    print(block.text)

        print()  # 空行分隔