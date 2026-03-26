"""
Tool Templates - 复制并按需定制这些工具模板。

每个工具需要两部分：
1. 定义（供模型读取的 JSON Schema）
2. 实现（对应的 Python 函数）
"""

from pathlib import Path   # 跨平台路径操作
import subprocess          # 执行 shell 子进程


WORKDIR = Path.cwd()  # 工作目录，默认为脚本运行所在目录


# =============================================================================
# 工具定义（填入 TOOLS 列表传给 API）
# =============================================================================

BASH_TOOL = {
    "name": "bash",
    "description": "Run a shell command. Use for: ls, find, grep, git, npm, python, etc.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"  # 要执行的 shell 命令字符串
            }
        },
        "required": ["command"],  # command 为必填参数
    },
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read file contents. Returns UTF-8 text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file"       # 相对于 WORKDIR 的文件路径
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read (default: all)"  # 可选：最多读取的行数，用于大文件截断
            },
        },
        "required": ["path"],  # limit 为可选参数
    },
}

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "Write content to a file. Creates parent directories if needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path for the file"   # 目标文件的相对路径
            },
            "content": {
                "type": "string",
                "description": "Content to write"            # 要写入的文本内容
            },
        },
        "required": ["path", "content"],  # 两个参数均为必填
    },
}

EDIT_FILE_TOOL = {
    "name": "edit_file",
    "description": "Replace exact text in a file. Use for surgical edits.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file"
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to find (must match precisely)"  # 需要被替换的原始文本，必须精确匹配
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text"  # 替换后的新文本
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
}

TODO_WRITE_TOOL = {
    "name": "TodoWrite",
    "description": "Update the task list. Use to plan and track progress.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "Complete list of tasks",  # 完整任务列表（全量覆盖，非增量追加）
                "items": {
                    "type": "object",
                    "properties": {
                        "content":    {"type": "string", "description": "Task description"},          # 任务描述文本
                        "status":     {"type": "string", "enum": ["pending", "in_progress", "completed"]},  # 任务状态枚举
                        "activeForm": {"type": "string", "description": "Present tense, e.g. 'Reading files'"},  # 进行时描述，用于进度展示
                    },
                    "required": ["content", "status", "activeForm"],  # 每条任务的三个字段均为必填
                },
            }
        },
        "required": ["items"],
    },
}

# Task 工具需依赖 AGENT_TYPES 动态生成，此处仅提供模板字符串供参考
TASK_TOOL_TEMPLATE = """
# Generate dynamically with agent types
TASK_TOOL = {
    "name": "Task",
    "description": f"Spawn a subagent for a focused subtask.\\n\\nAgent types:\\n{get_agent_descriptions()}",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short task name (3-5 words)"},
            "prompt": {"type": "string", "description": "Detailed instructions"},
            "agent_type": {"type": "string", "enum": list(AGENT_TYPES.keys())},
        },
        "required": ["description", "prompt", "agent_type"],
    },
}
"""


# =============================================================================
# 工具实现函数
# =============================================================================

def safe_path(p: str) -> Path:
    """
    安全性校验：确保路径不会逃逸出工作目录。
    防止形如 ../../../etc/passwd 的路径穿越攻击。
    """
    path = (WORKDIR / p).resolve()          # resolve() 展开所有 .. 和符号链接，得到绝对路径
    if not path.is_relative_to(WORKDIR):    # 若解析后路径不在 WORKDIR 内则拒绝
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    执行 shell 命令，附带安全检查。

    安全特性：
    - 拦截明显危险的命令关键字
    - 60 秒超时，防止命令挂起
    - 输出截断至 50KB，防止超大输出撑爆上下文
    """
    # 危险命令黑名单，匹配任意一项即拒绝执行
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            command,
            shell=True,          # 通过 shell 解释器执行，支持管道、通配符等
            cwd=WORKDIR,         # 固定工作目录
            capture_output=True, # 捕获 stdout/stderr，不直接打印
            text=True,           # 以字符串形式返回输出
            timeout=60           # 超时 60 秒
        )
        output = (result.stdout + result.stderr).strip()  # 合并两个输出流
        return output[:50000] if output else "(no output)"  # 截断并兜底空输出

    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


def run_read_file(path: str, limit: int = None) -> str:
    """
    读取文件内容，支持可选行数限制。

    特性：
    - 通过 safe_path 做路径安全校验
    - limit 参数可限制读取行数，避免大文件耗尽上下文
    - 整体输出截断至 50KB
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit]
            # 在末尾追加提示，告知模型还有多少行未显示
            lines.append(f"... ({len(text.splitlines()) - limit} more lines)")

        return "\n".join(lines)[:50000]  # 重新拼接并截断

    except Exception as e:
        return f"Error: {e}"


def run_write_file(path: str, content: str) -> str:
    """
    将内容写入文件，若父目录不存在则自动创建。

    特性：
    - 安全路径校验
    - parents=True 递归创建多级目录
    - 返回写入字节数，便于模型确认结果
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)  # 目录已存在时不报错
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    except Exception as e:
        return f"Error: {e}"


def run_edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    精确替换文件中的指定文本（外科手术式编辑）。

    特性：
    - 字符串精确匹配，不使用正则，避免误伤
    - replace(..., 1) 只替换第一处，防止批量误改
    - 找不到目标文本时明确报错，不静默失败
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()

        if old_text not in content:
            return f"Error: Text not found in {path}"  # 精确匹配失败时快速报错

        new_content = content.replace(old_text, new_text, 1)  # 第三个参数 1 = 只替换首次出现
        fp.write_text(new_content)
        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# 工具分发器
# =============================================================================

def execute_tool(name: str, args: dict) -> str:
    """
    将模型的工具调用分发到对应的实现函数。

    扩展方法：
    1. 在 TOOLS 列表中添加新工具的 JSON Schema 定义
    2. 编写对应的实现函数
    3. 在此处添加一个 if 分支即可接入
    """
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read_file(args["path"], args.get("limit"))  # limit 为可选，用 .get() 安全取值
    if name == "write_file":
        return run_write_file(args["path"], args["content"])
    if name == "edit_file":
        return run_edit_file(args["path"], args["old_text"], args["new_text"])
    # 在此添加更多工具分支...
    return f"Unknown tool: {name}"  # 兜底：工具名不匹配时返回错误提示