#!/usr/bin/env python3
"""
Agent Scaffold Script - 用于快速创建符合最佳实践的新 Agent 项目。

用法：
    python init_agent.py <agent-name> [--level 0-4] [--path <output-dir>]

示例：
    python init_agent.py my-agent                 # Level 1（4 个工具，默认）
    python init_agent.py my-agent --level 0      # 最简模式（仅 bash）
    python init_agent.py my-agent --level 2      # 含 TodoWrite 任务规划
    python init_agent.py my-agent --path ./bots  # 指定输出目录
"""

import argparse          # 命令行参数解析
import sys               # 用于 sys.exit() 异常退出
from pathlib import Path # 跨平台路径操作


# =============================================================================
# Agent 代码模板字典，key 为复杂度级别
# =============================================================================

TEMPLATES = {
    # ── Level 0：极简模式，仅 bash 一个工具（约 50 行）──────────────────────────
    0: '''#!/usr/bin/env python3
"""
Level 0 Agent - Bash is All You Need (~50 lines)

核心思想：单个 bash 工具足以完成一切任务。
子 Agent 通过自递归实现：python {name}.py "subtask"
"""

from anthropic import Anthropic
from dotenv import load_dotenv
import subprocess
import os

load_dotenv()  # 从 .env 文件加载环境变量

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL")
)
MODEL = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")

# 系统提示：告知模型用 bash 完成所有操作，子任务通过调用自身脚本实现
SYSTEM = """You are a coding agent. Use bash for everything:
- Read: cat, grep, find, ls
- Write: echo 'content' > file
- Subagent: python {name}.py "subtask"
"""

# 仅定义一个 bash 工具
TOOL = [{{
    "name": "bash",
    "description": "Execute shell command",
    "input_schema": {{"type": "object", "properties": {{"command": {{"type": "string"}}}}, "required": ["command"]}}
}}]

def run(prompt, history=[]):
    history.append({{"role": "user", "content": prompt}})
    while True:
        r = client.messages.create(model=MODEL, system=SYSTEM, messages=history, tools=TOOL, max_tokens=8000)
        history.append({{"role": "assistant", "content": r.content}})
        # 无工具调用时返回最终文本
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if hasattr(b, "text"))
        results = []
        for b in r.content:
            if b.type == "tool_use":
                print(f"> {{b.input['command']}}")  # 打印正在执行的命令
                try:
                    out = subprocess.run(b.input["command"], shell=True, capture_output=True, text=True, timeout=60)
                    output = (out.stdout + out.stderr).strip() or "(empty)"
                except Exception as e:
                    output = f"Error: {{e}}"
                results.append({{"type": "tool_result", "tool_use_id": b.id, "content": output[:50000]}})
        history.append({{"role": "user", "content": results}})

if __name__ == "__main__":
    h = []
    print("{name} - Level 0 Agent\\nType 'q' to quit.\\n")
    # 海象运算符 := 在条件判断中同时赋值，简化循环写法
    while (q := input(">> ").strip()) not in ("q", "quit", ""):
        print(run(q, h), "\\n")
''',

    # ── Level 1：基础模式，4 个核心工具（约 200 行）────────────────────────────
    1: '''#!/usr/bin/env python3
"""
Level 1 Agent - Model as Agent (~200 lines)

核心思想：4 个工具覆盖 90% 的编码任务。
模型本身就是 Agent，代码只负责运行循环。
"""

from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
import subprocess
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL")
)
MODEL = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
WORKDIR = Path.cwd()  # 工作目录固定为脚本运行所在目录

SYSTEM = f"""You are a coding agent at {{WORKDIR}}.

Rules:
- Prefer tools over prose. Act, don't just explain.
- Never invent file paths. Use ls/find first if unsure.
- Make minimal changes. Don't over-engineer.
- After finishing, summarize what changed."""

# 4 个核心工具：bash / 读文件 / 写文件 / 精确编辑
TOOLS = [
    {{"name": "bash", "description": "Run shell command",
     "input_schema": {{"type": "object", "properties": {{"command": {{"type": "string"}}}}, "required": ["command"]}}}},
    {{"name": "read_file", "description": "Read file contents",
     "input_schema": {{"type": "object", "properties": {{"path": {{"type": "string"}}}}, "required": ["path"]}}}},
    {{"name": "write_file", "description": "Write content to file",
     "input_schema": {{"type": "object", "properties": {{"path": {{"type": "string"}}, "content": {{"type": "string"}}}}, "required": ["path", "content"]}}}},
    {{"name": "edit_file", "description": "Replace exact text in file",
     "input_schema": {{"type": "object", "properties": {{"path": {{"type": "string"}}, "old_text": {{"type": "string"}}, "new_text": {{"type": "string"}}}}, "required": ["path", "old_text", "new_text"]}}}},
]

def safe_path(p: str) -> Path:
    """路径安全校验：防止 ../../../etc/passwd 类路径穿越攻击。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {{p}}")
    return path

def execute(name: str, args: dict) -> str:
    """工具分发器：根据工具名调用对应实现，返回字符串结果。"""
    if name == "bash":
        # 危险命令黑名单拦截
        dangerous = ["rm -rf /", "sudo", "shutdown", "> /dev/"]
        if any(d in args["command"] for d in dangerous):
            return "Error: Dangerous command blocked"
        try:
            r = subprocess.run(args["command"], shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=60)
            return (r.stdout + r.stderr).strip()[:50000] or "(empty)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (60s)"
        except Exception as e:
            return f"Error: {{e}}"

    if name == "read_file":
        try:
            return safe_path(args["path"]).read_text()[:50000]  # 截断至 50KB
        except Exception as e:
            return f"Error: {{e}}"

    if name == "write_file":
        try:
            p = safe_path(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)  # 自动创建父目录
            p.write_text(args["content"])
            return f"Wrote {{len(args['content'])}} bytes to {{args['path']}}"
        except Exception as e:
            return f"Error: {{e}}"

    if name == "edit_file":
        try:
            p = safe_path(args["path"])
            content = p.read_text()
            if args["old_text"] not in content:
                return f"Error: Text not found in {{args['path']}}"
            p.write_text(content.replace(args["old_text"], args["new_text"], 1))  # 只替换首次出现
            return f"Edited {{args['path']}}"
        except Exception as e:
            return f"Error: {{e}}"

    return f"Unknown tool: {{name}}"

def agent(prompt: str, history: list = None) -> str:
    """Agent 主循环：持续调用工具直到模型不再触发工具调用。"""
    if history is None:
        history = []
    history.append({{"role": "user", "content": prompt}})

    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=history, tools=TOOLS, max_tokens=8000
        )
        history.append({{"role": "assistant", "content": response.content}})

        # 停止原因不是工具调用 → 任务完成，提取并返回纯文本
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"> {{block.name}}: {{str(block.input)[:100]}}")  # 调试输出：工具名 + 参数预览
                output = execute(block.name, block.input)
                print(f"  {{output[:100]}}...")                          # 调试输出：结果前 100 字符
                results.append({{"type": "tool_result", "tool_use_id": block.id, "content": output}})
        history.append({{"role": "user", "content": results}})  # 工具结果回填历史，供下轮模型读取

if __name__ == "__main__":
    print(f"{name} - Level 1 Agent at {{WORKDIR}}")
    print("Type 'q' to quit.\\n")
    h = []
    while True:
        try:
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C 优雅退出
            break
        if query in ("q", "quit", "exit", ""):
            break
        print(agent(query, h), "\\n")
''',
}

# .env.example 模板，创建项目时自动生成，用户按需填写真实 Key
ENV_TEMPLATE = '''# API Configuration
ANTHROPIC_API_KEY=sk-xxx
ANTHROPIC_BASE_URL=https://api.anthropic.com
MODEL_NAME=claude-sonnet-4-20250514
'''


def create_agent(name: str, level: int, output_dir: Path):
    """在指定目录下创建新的 Agent 项目，包含代码文件、环境变量示例和 .gitignore。"""

    # 校验 level：2-4 尚未在脚手架中实现，需从完整仓库复制
    if level not in TEMPLATES and level not in (2, 3, 4):
        print(f"Error: Level {level} not yet implemented in scaffold.")
        print("Available levels: 0 (minimal), 1 (4 tools)")
        print("For levels 2-4, copy from mini-claude-code repository.")
        sys.exit(1)

    # 以 agent 名称创建子目录（已存在则忽略）
    agent_dir = output_dir / name
    agent_dir.mkdir(parents=True, exist_ok=True)

    # 写入 Agent 主文件，用 name 填充模板中的 {name} 占位符
    agent_file = agent_dir / f"{name}.py"
    template = TEMPLATES.get(level, TEMPLATES[1])  # 未找到对应 level 时回退到 Level 1
    agent_file.write_text(template.format(name=name))
    print(f"Created: {agent_file}")

    # 写入 .env.example，提醒用户复制并填写真实 API Key
    env_file = agent_dir / ".env.example"
    env_file.write_text(ENV_TEMPLATE)
    print(f"Created: {env_file}")

    # 写入 .gitignore，防止 .env（含真实密钥）和编译缓存被提交到版本控制
    gitignore = agent_dir / ".gitignore"
    gitignore.write_text(".env\n__pycache__/\n*.pyc\n")
    print(f"Created: {gitignore}")

    # 打印后续操作指引
    print(f"\nAgent '{name}' created at {agent_dir}")
    print(f"\nNext steps:")
    print(f"  1. cd {agent_dir}")
    print(f"  2. cp .env.example .env")
    print(f"  3. Edit .env with your API key")
    print(f"  4. pip install anthropic python-dotenv")
    print(f"  5. python {name}.py")


def main():
    # 配置命令行参数解析器，RawDescriptionHelpFormatter 保留 epilog 的缩进格式
    parser = argparse.ArgumentParser(
        description="Scaffold a new AI coding agent project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Levels:
  0  Minimal (~50 lines) - Single bash tool, self-recursion for subagents
  1  Basic (~200 lines)  - 4 core tools: bash, read, write, edit
  2  Todo (~300 lines)   - + TodoWrite for structured planning
  3  Subagent (~450)     - + Task tool for context isolation
  4  Skills (~550)       - + Skill tool for domain expertise
        """
    )
    # 必填位置参数：Agent 名称
    parser.add_argument("name", help="Name of the agent to create")
    # 可选参数：复杂度级别，限定为 0-4 的整数，默认 1
    parser.add_argument("--level", type=int, default=1, choices=[0, 1, 2, 3, 4],
                       help="Complexity level (default: 1)")
    # 可选参数：输出目录，默认为当前目录
    parser.add_argument("--path", type=Path, default=Path.cwd(),
                       help="Output directory (default: current directory)")

    args = parser.parse_args()
    create_agent(args.name, args.level, args.path)


if __name__ == "__main__":
    main()