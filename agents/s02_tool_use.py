#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach.
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载当前目录下的 .env 文件（覆盖已有环境变量）
load_dotenv(override=True)

# 从环境变量中读取模型 ID
WORKDIR = Path.cwd()
# 创建 Anthropic 客户端（可连接官方或自定义 endpoint）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词（格式化字符串：执行 os.getcwd()，然后把结果插进字符串里）
SYSTEM = f"You are a coding agent at {WORKDIR}. Use bash to solve tasks. Act, don't explain."



# 安全地解析路径，防止越界访问
def safe_path(p: str) -> Path: # "-> Path" 是类型提示
    path = (WORKDIR / p).resolve() # pathlib 中的特殊拼接逻辑
    # 验证生成的路径是否仍然在 WORKDIR 范围内：
    if not path.is_relative_to(WORKDIR): 
        raise ValueError(f"Path escapes workspace: {p}")
    return path


# 执行 bash 命令的函数, 带有基础安全保护（防止误删系统等）
def run_bash(command: str) -> str:
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
            cwd=WORKDIR,  # 当前工作目录
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



# 安全地读取文件内容（限制路径、长度）
def run_read(path: str, limit: int = None) -> str:

    # 先通过 safe_path 确认路径合法，再读取文本内容：
    text = safe_path(path).read_text() 

    lines = text.splitlines() 
    # 如果设置了行数限制，并且文件行数超过限制，则只返回前 limit 行：
    if limit and limit < len(lines): 
        lines = lines[:limit]
    
     # 长度硬限制，将返回的字符串强制截断在 50,000 个字符以内：
    return "\n".join(lines)[:50000]



# 安全地写入文件内容（限制路径）
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"



# 安全地编辑文件内容（限制路径）
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具列表（定义工具的名称、描述和输入格式）。模型会根据这个列表来决定调用哪个工具，以及如何构造输入参数。
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]
# 上述工具的执行手册：
# -- The dispatch map: {tool_name: handler} --
"""
分发器模式的字典直接把指令名作为键(Key),把处理逻辑作为值(Value)。
当你收到一个任务时,去kw变量里找 path 和 limit ，然后交给 run_read 。为了便于理解,我给kw都改了改。
每个 lambda 都是一个独立的小世界(匿名函数)。所以假如全用kw也没问题。
"""
TOOL_HANDLERS = {
    "bash":       lambda **kw_bash: run_bash(kw_bash["command"]),
    "read_file":  lambda **pkg_read: run_read(pkg_read["path"], pkg_read.get("limit")),
    "write_file": lambda **data_in: run_write(data_in["path"], data_in["content"]),
    "edit_file":  lambda **kw_edit: run_edit(kw_edit["path"], kw_edit["old_text"], kw_edit["new_text"]),
}



# -- 核心 Agent 循环 --
def agent_loop(messages: list):
    while True:
        # 调用 LLM（带上下文 + 工具定义）
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 把模型的回复加入对话历史
        messages.append({"role": "assistant","content": response.content})

        # 如果模型没有调用工具 -> 结束循环
        if response.stop_reason != "tool_use":
            return

        # 否则：执行工具调用
        results = []

        # 循环中按名称查找处理函数。response.content 是一个 block 列表
        for block in response.content:

            # 如果这个 block 是工具调用
            if block.type == "tool_use":
                # 去 TOOL_HANDLERS 字典里查找, 返回一个lambda函数的包装
                handler = TOOL_HANDLERS.get(block.name)
                # **block.input：AI 提供的参数被解包传入handler。如果工具名不存在(else)，返回错误字符串。
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                # 打印命令与部分输出
                print(f"> {block.name}: {output[:200]}")
                # 收集结果
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # 对应 tool_use 的 ID
                    "content": output
                })

        # 把工具执行结果作为“用户消息”喂回模型
        messages.append({"role": "user","content": results})


# 程序入口
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):             # Ctrl+D 或 Ctrl+C 退出
            break
        if query.strip().lower() in ("q", "exit", ""):    # q / exit / 空字符串 -> 退出
            break
        history.append({"role": "user","content": query}) # 把用户输入加入历史

        # 运行 agent 循环
        agent_loop(history)

        # 取最后一条消息（模型输出）
        response_content = history[-1]["content"]
        # 如果是结构化 block（列表）
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"): # 有 text 属性就打印
                    print(block.text)

        print()  # 空行分隔
