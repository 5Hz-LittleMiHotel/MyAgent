#!/usr/bin/env python3
# Harness: on-demand knowledge -- domain expertise, loaded when the model asks.
"""
s05_skill_loading.py - Skills

Two-layer skill injection that avoids bloating the system prompt:

    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)
    Layer 2 (on demand): full skill body in tool_result

    skills/
      pdf/
        SKILL.md          <-- frontmatter (name, description) + body
      code-review/
        SKILL.md

    System prompt:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - pdf: Process PDF files...        |  <-- Layer 1: metadata only
    |   - code-review: Review code...      |
    +--------------------------------------+

    When model calls load_skill("pdf"):
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   Full PDF processing instructions   |  <-- Layer 2: full body
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                             |
    +--------------------------------------+

Key insight: "Don't put everything in the system prompt. Load on demand."
"""

import os
import re
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"



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
]


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw_bash: run_bash(kw_bash["command"]),
    "read_file":  lambda **pkg_read: run_read(pkg_read["path"], pkg_read.get("limit")),
    "write_file": lambda **data_in: run_write(data_in["path"], data_in["content"]),
    "edit_file":  lambda **kw_edit: run_edit(kw_edit["path"], kw_edit["old_text"], kw_edit["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}


# %% ---------------- Agent loop with nag reminder injection ----------------

def agent_loop(messages: list):
    rounds_since_todo = 0 #  in_process 计数器
    while True:
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

        # 循环中按名称查找处理函数
        for block in response.content:
            # 如果这个 block 是工具调用
            if block.type == "tool_use":
                # 去 TOOL_HANDLERS 字典里查找, 返回一个lambda函数的包装
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

        history.append({"role": "user","content": query})
        agent_loop(history)

        response_content = history[-1]["content"]
        # 如果是结构化 block(列表)
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"): # 有 text 属性就打印
                    print(block.text)

        print()  # 空行分隔

"""
现在有一个问题: 
设置了一个SkillLoader类用于从skills目录加载工具, 并设置两层加载, Layer 1: 摘要, Layer 2: 按需加载全文。
一旦设置这个两层加载机制, 好像s04的一些代码就显得冗余了。

第一个agent-builder的SKILL.md包含的提示词内容已简化, 下面是该md文件提供的资源: 
## Resources

**Philosophy & Theory**:
- `references/agent-philosophy.md` - Deep dive into why agents work

**Implementation**:
- `references/minimal-agent.py` - Complete working agent (~80 lines)
- `references/tool-templates.py` - Capability definitions
- `references/subagent-pattern.py` - Context isolation

**Scaffolding**:
- `scripts/init_agent.py` - Generate new agent projects


第二个code-review的SKILL.md包含的提示词内容已简化: 
只有当用户提到‘审查’、‘Bug’、‘安全’这些词时，才需要激活这个模式。包含的内容大概有：
Node.js	npm audit	检查 package.json 依赖的安全漏洞
Python	pip-audit	检查 Python 依赖的安全漏洞
Rust	cargo audit	检查 Rust crate 依赖的安全漏洞

当读到第三个 mcp-builder 的 SKILL.md ，智能体将会做：
依据此文件编写Python或TypeScript代码，构建独立的MCP服务器以集成外部API、数据库或本地资源。
它会定义工具函数、配置启动命令，并指导用户将其注册到配置文件中，从而动态扩展自身能力以执行特定任务。

第四个 pdf 的 SKILL.md ：
该文件定义了PDF处理技能，涵盖读取、创建、合并及拆分等操作，并推荐了PyMuPDF等库。
Agent读到后，将依据用户指令调用相应工具或代码，执行如提取文本、生成报告或整理文档等具体任务。
"""


"""
而且选择tesk还是SKILL建立agent, 也需要考虑一些因素:

task 工具是运行时能力——LLM 在 agent loop 执行过程中，可以动态决定"我现在需要隔离一个子任务"，
立刻 fork 出一个干净上下文的子代理去执行，结果直接返回到当前对话流里继续推进。

SKILL 里的 subagent 是知识/模板——它告诉 LLM "如何构建一个子代理"，但 LLM 要真正用它，必须先
 load_skill，读懂内容，然后自己用 write_file + bash 把代码写到磁盘，再 bash 去执行一个新进程。

如果删掉 task, LLM 想隔离子任务: 
  → load_skill("agent-builder")
  → 读 SKILL.md，理解模板
  → write_file("subtask_runner.py", ...)   # 写出一个脚本
  → bash("python subtask_runner.py")       # 起一个新进程
  → 等待进程结束，解析输出

相比不删除：
LLM 想隔离子任务
  → task(prompt="...")    # 一步完成
多出来的不只是步骤数——新进程方案有几个真实代价：
没有共享内存里的 TODO 状态、每次都要重新初始化 Anthropic client、
子进程的输出需要自己设计序列化格式、出错了没有统一的异常捕获。

什么时候删 task 才合理: 
如果你未来想让子代理真正并发执行（asyncio 或多进程），或者需要子代理跨机器运行，
那时候 SKILL 里的外部进程方案才有优势，task 工具确实可以退役。但在现在的单线程同步架构里，
task 是最轻量、最可靠的隔离机制。

所以这里可以设置系统提示词, 让大模型来选择task or 新建agent工厂. 
可以后期优化一下. 反正这个版本是没有task的, 给删了.
"""