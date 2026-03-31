#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.

import json
import os
import re
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3

# %% -- SkillLoader: scan skills/<name>/SKILL.md with YAML frontmatter --
class SkillLoader:
    def __init__(self, skills_dir: Path):
        # skills_dir: 技能目录路径(例如 ./skills)
        self.skills_dir = skills_dir

        # 存储所有加载后的技能: 
        # {
        #   "skill_name": {
        #       "meta": {...},   # frontmatter 元数据
        #       "body": "...",   # 技能正文
        #       "path": "..."    # 文件路径
        #   }
        # }
        self.skills = {}

        # 初始化时自动加载所有技能
        self._load_all()

    def _load_all(self):
        # 如果技能目录不存在, 直接返回(避免报错)
        if not self.skills_dir.exists():
            return

        # 递归查找所有名为 SKILL.md 的文件
        # rglob 会遍历子目录
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            # 读取文件内容
            text = f.read_text()

            # 解析 frontmatter(元信息)和正文
            meta, body = self._parse_frontmatter(text)

            # 技能名称优先取 meta 中的 name
            # 如果没有, 则使用文件所在目录名
            name = meta.get("name", f.parent.name)

            # 存入 skills 字典
            self.skills[name] = {
                "meta": meta,
                "body": body,
                "path": str(f)
            }

    def _parse_frontmatter(self, text: str) -> tuple:
        """
        解析 Markdown 文件中的 frontmatter(YAML 风格元数据)

        格式示例: 
        ---
        name: git_commit
        description: Rules for writing commit messages
        tags: git, workflow
        ---
        这里是正文内容...
        """

        # 使用正则匹配 frontmatter(--- 包裹的部分)
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)

        # 如果没有 frontmatter, 返回空 meta + 全部文本作为正文
        if not match:
            return {}, text

        meta = {}

        # 逐行解析 YAML(简化版, 只支持 key: value)
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()

        # 返回: 
        # - meta(字典)
        # - body(去掉 frontmatter 后的正文)
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """
        Layer 1: 提供“技能摘要”, 用于 system prompt

        只返回: 
        - 名称
        - 简短描述
        - tags(可选)

        目的: 让模型知道“有哪些技能可用”, 但不占太多 token
        """

        # 如果没有技能, 返回占位信息
        if not self.skills:
            return "(no skills available)"

        lines = []

        # 遍历所有技能
        for name, skill in self.skills.items():
            # description 用于简要说明技能用途
            desc = skill["meta"].get("description", "No description")

            # tags 用于辅助检索(例如: git, testing)
            tags = skill["meta"].get("tags", "")

            # 构造一行描述
            line = f"  - {name}: {desc}"

            # 如果有 tags, 则附加
            if tags:
                line += f" [{tags}]"

            lines.append(line)

        # 多行拼接为字符串(供 system prompt 使用)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """
        Layer 2: 返回完整技能内容(用于 tool_result 注入)

        👉 只有在模型明确请求某个 skill 时才调用
        👉 避免把所有技能塞进 prompt(节省 token)
        """

        # 查找对应技能, 如果不存在, 返回错误 + 可用技能列表(帮助模型纠错)
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"

        # 返回结构化内容(用 XML-like 包裹, 方便模型解析)
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


# 全局实例(通常在程序启动时初始化一次)
SKILL_LOADER = SkillLoader(SKILLS_DIR)


# 系统提示词
SYSTEM = f"""
You are a coding agent at {WORKDIR}.
Planning: Use the todo tool to plan multi-step tasks (mark as in_progress when starting, completed when done). Use the task tool to delegate exploration or subtasks.
Knowledge: Use load_skill to access specialized knowledge for unfamiliar topics. Available skills: {SKILL_LOADER.get_descriptions()}.
Behavior: Prefer using tools over generating prose.
"""
SUBAGENT_SYSTEM = f"""You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."""

# %% ------------------------- compression -------------------------

def estimate_tokens(messages: list) -> int:
    """
    粗略估算 token 数量：
    假设 1 token ≈ 4 个字符
    用于判断是否需要进行上下文压缩
    """
    return len(str(messages)) // 4


def extract_critical(messages):
    # 找最近一条报错、最近一条in_progress的todo
    errors = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    content = part.get("content", "")
                    if "Error" in content or "Traceback" in content:
                        errors.append(content[-2000:])  # 只保留尾部
    return errors[-1] if errors else None


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    """
    第一层压缩（轻量级）：
    - 找到历史中的 tool_result
    - 只保留最近 KEEP_RECENT 条
    - 更早的结果替换为占位符（减少 token）
    """

    # 收集所有 tool_result 的位置：
    # (消息索引, 内容索引, tool_result 本体)
    tool_results = []
    for msg_idx, msg in enumerate(messages):

        # tool_result 总是在 user role 且 content 是 list
        if msg["role"] == "user" and isinstance(msg.get("content"), list):

            # 遍历 content 中的每一项（block）
            for part_idx, part in enumerate(msg["content"]):

                # 找到 tool_result 类型
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))

    # 如果 tool_result 数量不多，不做压缩
    if len(tool_results) <= KEEP_RECENT:
        return messages

    # ---------------------------
    # 构建 tool_use_id → tool_name 映射
    # ---------------------------
    tool_name_map = {}

    for msg in messages:
        if msg["role"] == "assistant":

            content = msg.get("content", [])

            # assistant 的 content 是 block 列表
            if isinstance(content, list):
                for block in content:

                    # 找到 tool_use block
                    if hasattr(block, "type") and block.type == "tool_use":

                        # 建立映射：tool_use_id → 工具名
                        tool_name_map[block.id] = block.name

    # ---------------------------
    # 清理旧的 tool_result（只保留最近 KEEP_RECENT 条）
    # ---------------------------
    to_clear = tool_results[:-KEEP_RECENT]

    for _, _, result in to_clear:

        # 只处理较长内容（避免无意义替换）
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:

            tool_id = result.get("tool_use_id", "")

            # 找到对应工具名（如果找不到就标 unknown）
            tool_name = tool_name_map.get(tool_id, "unknown")

            # 替换为简短占位符（极大减少 token）
            result["content"] = f"[Previous: used {tool_name}]"

    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    """
    第二层压缩（重压缩）：
    1. 把完整对话保存到磁盘（防止信息丢失）
    2. 让 LLM 生成摘要
    3. 用摘要替换整个 history
    """

    # ---------------------------
    # 保存完整对话到本地文件（jsonl）
    # ---------------------------
    TRANSCRIPT_DIR.mkdir(exist_ok=True)

    # 生成带时间戳的文件名
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"

    with open(transcript_path, "w") as f:
        for msg in messages:
            # 每条消息一行（jsonl 格式）
            f.write(json.dumps(msg, default=str) + "\n")

    print(f"[transcript saved: {transcript_path}]")

    # ---------------------------
    # 构造用于总结的文本（截断防止超长）
    # ---------------------------
    conversation_text = json.dumps(messages, default=str)[:80000]

    # 调用 LLM 生成摘要
    response = client.messages.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content":
                "Summarize this conversation for continuity. Include: "
                "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                "Be concise but preserve critical details.\n\n"
                + conversation_text
        }],
        max_tokens=2000,
    )

    # 取出摘要文本（第一个 text block）
    summary = response.content[0].text

    # 如果找到了报错内容，就把它包在<critical_context>标签里
    critical = extract_critical(messages)
    preserved = f"\n\n<critical_context>\n{critical}\n</critical_context>" if critical else ""

    # ---------------------------
    # 用摘要替换整个对话历史
    # ---------------------------
    # LLM生成的摘要, 最近一条报错的尾部的2000个字符, 以及现在的状态.
    return [
        # 摘要负责还原"做了什么"，critical_context(preserved)负责保留"卡在哪里"，todos负责告诉模型"现在状态是什么"。
        # 三者互补，弥补auto_compact丢失推理证据的问题
        {
            "role": "user",
            "content": (
                f"[Conversation compressed. Transcript: {transcript_path}]\n\n"
                f"{summary}"
                f"{preserved}\n\n"
                f"<todos>\n{TODO.render()}\n</todos>"
            )
        },
        {
            "role": "assistant",
            "content": "Understood. I have the context from the summary. Continuing."
        },
    ]


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
            # 每个任务必须包含内容
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            # 任务状态
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            # 限定 in_progress 数量
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        
        self.items = validated
        # 转换成 ASCII 文本界面
        return self.render()

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



# %% ------------ Tool implementations(sandbox) ------------

# 安全地解析路径, 防止越界访问
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR): 
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    # 定义并剔除危险命令
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, 
                           capture_output=True,  # 捕获 stdout/stderr
                           text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)" # 超时保护



# %% ---------------- Tool implementations ----------------

# 安全地读取文件内容
def run_read(path: str, limit: int = None) -> str:
    try:
        # 先通过 safe_path 确认路径合法, 再读取文本内容: 
        lines = safe_path(path).read_text() .splitlines() 
        # 如文件行数超过限制, 则只返回前 limit 行, 并告诉LLM还有多少行没给它看: 
        if limit and limit < len(lines): 
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        # 长度硬限制, 将返回的字符串强制截断在 50,000 个字符以内: 
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


# 安全地写入文件内容
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


# 安全地编辑文件内容(限制路径)
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



# 工具列表(定义工具的名称、描述和输入格式)。模型会根据这个列表来决定调用哪个工具, 以及如何构造输入参数。
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw_bash: run_bash(kw_bash["command"]),
    "read_file":  lambda **pkg_read: run_read(pkg_read["path"], pkg_read.get("limit")),
    "write_file": lambda **data_in: run_write(data_in["path"], data_in["content"]),
    "edit_file":  lambda **kw_edit: run_edit(kw_edit["path"], kw_edit["old_text"], kw_edit["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":    lambda **kw: "Manual compression requested.",
}


# %% ---------------- Agent loop with nag reminder injection ----------------

def agent_loop(messages: list):
    rounds_since_todo = 0 #  in_process 计数器
    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        # 调用 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 把模型的回复加入对话历史
        messages.append({"role": "assistant","content": response.content})
        # 如果模型没有调用工具 -> 结束循环
        if response.stop_reason != "tool_use":
            return
        # 否则: 执行工具调用
        results = []
        used_todo = False
        manual_compact = False
        # 循环中按名称查找处理函数
        for block in response.content:
            # 如果这个 block 是工具调用
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        # **block.input: AI 提供的参数被解包传入handler。如果工具名不存在(else), 返回错误字符串。
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
        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)



# %% ---------------- main ----------------
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user","content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list): # 如果是结构化 block(列表)
            for block in response_content:
                if hasattr(block, "text"): # 有 text 属性就打印
                    print(block.text)
        print()  # 空行分隔