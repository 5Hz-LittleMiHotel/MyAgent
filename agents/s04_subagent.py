#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.

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

# 系统提示词（告诉LLM使用todo工具来计划多步骤任务。开始前标记in_progress，完成后标记completed）
SYSTEM = f"""You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks.
If necessary, use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""
SUBAGENT_SYSTEM = f"""You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."""


# %% ------- TodoManager: structured state the LLM writes to -------
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        # 强调任务列表不能超过 20 项
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0 # 初始化正在执行的任务的数量

        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            # 每个任务必须包含 text(内容), 不能传一个空任务
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            
            # 任务状态只能是这三个之一(待办，进行中，已完成)：
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            
            # 如果 LLM 试图把两个任务都标记为 in_progress, 系统会报ValueError
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        
        self.items = validated

        # 转换成 ASCII 文本界面
        return self.render()

    """
    将内存中的列表对象转换成人类(和LLM)都能直观阅读的 ASCII 文本界面
    [ ] 代表 pending
    [>] 代表 in_progress(正在处理)
    [x] 代表 completed
    """
    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", 
                      "in_progress": "[>]",
                      "completed": "[x]"
                    }[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

TODO = TodoManager()



# %% ---------------------- sandbox ----------------------

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
    
    try: # 执行命令
        r = subprocess.run(command, shell=True, cwd=WORKDIR, 
                           capture_output=True,  # 捕获 stdout/stderr
                           text=True, timeout=120)
        # 合并标准输出和错误输出
        out = (r.stdout + r.stderr).strip()
        # 限制输出长度（防止爆内存）
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)" # 超时保护



# %% ---------------- Tool implementations ----------------

# 安全地读取文件内容（限制路径、长度）
def run_read(path: str, limit: int = None) -> str:
    try:
        # 先通过 safe_path 确认路径合法，再读取文本内容：
        lines = safe_path(path).read_text() .splitlines() 
        # 如果设置了行数限制，并且文件行数超过限制，则只返回前 limit 行, 并告诉LLM还有多少行没给它看：
        if limit and limit < len(lines): 
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        # 长度硬限制，将返回的字符串强制截断在 50,000 个字符以内：
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"



# 安全地写入文件内容（限制路径）
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes" # to {path}": 输入参数里已经包含了path。减少 AI 的认知负荷。
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



# -- TODO: Subagent: fresh context, filtered tools, summary-only return --
def handle_task(prompt: str, description: str = "subtask") -> str:
    print(f"> task ({description}): {prompt[:80]}")
    output = run_subagent(prompt)
    return output

def run_subagent(prompt: str) -> str:
    """
    运行一个“子代理”

    特点：
    - 有自己独立的上下文 sub_messages
    - 可以调用工具
    - 最终只返回“总结结果”给父 agent
    """

    # 初始化子代理的对话历史（和主 agent 完全隔离）
    # 这里只放一条用户输入（prompt）
    sub_messages = [{
        "role": "user",
        "content": prompt
    }]  # fresh context（全新上下文）

    # 限制最多循环 30 次（防止死循环或模型失控）
    for _ in range(30):  # safety limit

        # 调用 LLM（子代理专用 system prompt + 工具）
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,   # 子代理的系统提示词（通常更专注）
            messages=sub_messages,   # 子代理自己的历史
            tools=CHILD_TOOLS,       # 子代理可用工具（通常是子集）
            max_tokens=8000,
        )

        # 把模型回复加入子代理历史（assistant 角色）
        sub_messages.append({
            "role": "assistant",
            "content": response.content
        })

        # 如果模型没有调用工具（说明任务完成 or 不需要工具）
        # 就退出循环
        if response.stop_reason != "tool_use":
            break

        # 用于收集所有工具执行结果
        results = []

        # 遍历模型返回的 block 列表
        for block in response.content:

            # 如果是工具调用 block
            if block.type == "tool_use":

                # 根据工具名找到对应的处理函数
                handler = CHILD_TOOL_HANDLERS.get(block.name)

                # 执行工具：
                # - 如果有 handler，就调用它
                # - 如果没有，返回错误信息
                output = (
                    handler(**block.input)
                    if handler
                    else f"Unknown tool: {block.name}"
                )

                # 把工具执行结果封装成 tool_result
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # 对应 tool_use 的 ID
                    "content": str(output)[:50000]  # 限制输出长度
                })

        # 把工具结果作为“用户消息”喂回模型（继续循环）
        sub_messages.append({
            "role": "user",
            "content": results
        })

    # ---------------------------
    # 循环结束：返回结果给父 agent
    # ---------------------------

    # 从最后一次 response 中提取所有 text block
    # 并拼接成一个字符串
    return "".join(
        b.text for b in response.content
        if hasattr(b, "text")  # 只取 text 类型 block
    ) or "(no summary)"  # 如果没有文本，就返回默认值



# 工具列表（定义工具的名称、描述和输入格式）。模型会根据这个列表来决定调用哪个工具，以及如何构造输入参数。

CHILD_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]
PARENT_TOOLS = CHILD_TOOLS + [
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
    {"name": "task", "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
     "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}}, "required": ["prompt"]}},
]
# -- The dispatch map: {tool_name: handler} --
"""
分发器模式的字典直接把指令名作为键(Key),把处理逻辑作为值(Value)。
当你收到一个任务时,去kw变量里找 path 和 limit ，然后交给 run_read 。为了便于理解,我给kw都改了改。
每个 lambda 都是一个独立的匿名函数。所以假如全用kw也没问题。
"""
CHILD_TOOL_HANDLERS = {
    "bash":       lambda **kw_bash: run_bash(kw_bash["command"]),
    "read_file":  lambda **pkg_read: run_read(pkg_read["path"], pkg_read.get("limit")),
    "write_file": lambda **data_in: run_write(data_in["path"], data_in["content"]),
    "edit_file":  lambda **kw_edit: run_edit(kw_edit["path"], kw_edit["old_text"], kw_edit["new_text"]),
}
PARENT_TOOLS_HANDLERS = {
    **CHILD_TOOL_HANDLERS,
    "todo":       lambda **kw: TODO.update(kw["items"]),
    "task":       lambda **kw: handle_task(kw["prompt"], kw.get("description", "subtask")),
}


# %% ---------------- Agent loop with nag reminder injection ----------------
# TODO: 既然不是所有的智能体都需要todo工具了，需要对agent_loop进行改造
def agent_loop(messages: list):
    rounds_since_todo = 0 # 初始化计数器
    
    while True:

        # 调用 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=PARENT_TOOLS, max_tokens=8000,
        )

        # 把模型的回复加入对话历史
        messages.append({"role": "assistant","content": response.content})

        # 如果模型没有调用工具 -> 结束循环
        if response.stop_reason != "tool_use":
            return

        # 否则：执行工具调用
        results = []
        used_todo = False

        # 循环中按名称查找处理函数
        for block in response.content:

            # 如果这个 block 是工具调用
            if block.type == "tool_use":
                # 去 TOOL_HANDLERS 字典里查找, 返回一个lambda函数的包装
                handler = PARENT_TOOLS_HANDLERS.get(block.name)
                
                # 加入了一个捕获异常的手段
                try:
                    # **block.input：AI 提供的参数被解包传入handler。如果工具名不存在(else)，返回错误字符串。
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                
                # 打印命令与部分输出
                print(f"> {block.name}: {output[:200]}")
                # 收集结果
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # 对应 tool_use 的 ID
                    "content": output
                })

                if block.name == "todo":
                    used_todo = True

        # 模型连续 3 轮以上不调用 todo 时注入提醒。
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        
        # 把工具执行结果作为“用户消息”喂回模型
        messages.append({"role": "user","content": results})



# %% ---------------- main ----------------
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

        agent_loop(history)

        # 取最后一条消息
        response_content = history[-1]["content"]
        # 如果是结构化 block（列表）
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"): # 有 text 属性就打印
                    print(block.text)

        print()  # 空行分隔