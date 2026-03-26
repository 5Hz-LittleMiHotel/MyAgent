"""
Subagent Pattern - 如何实现 Task 工具以实现上下文隔离。

核心思想：派生携带**独立上下文**的子 Agent，避免"上下文污染"——
即探索过程中的细节把主对话的 token 窗口撑满。
"""

import time  # 用于计算子任务耗时
import sys   # 用于 sys.stdout.write 实现同行进度刷新

# 假设 client、MODEL、execute_tool 已在其他地方定义


# =============================================================================
# AGENT 类型注册表
# =============================================================================

AGENT_TYPES = {
    # explore：只读，用于搜索和分析
    "explore": {
        "description": "Read-only agent for exploring code, finding files, searching",
        "tools": ["bash", "read_file"],  # 无写权限！
        "prompt": "You are an exploration agent. Search and analyze, but NEVER modify files. Return a concise summary of what you found.",
    },

    # code：全权限，用于实际编码实现
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",  # 拥有所有工具
        "prompt": "You are a coding agent. Implement the requested changes efficiently. Return a summary of what you changed.",
    },

    # plan：只读，用于设计方案
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],  # 只读
        "prompt": "You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make any changes.",
    },

    # 可在此添加自定义类型...
    # "test": {
    #     "description": "Testing agent for running and analyzing tests",
    #     "tools": ["bash", "read_file"],
    #     "prompt": "Run tests and report results. Don't modify code.",
    # },
}


def get_agent_descriptions() -> str:
    """生成 Task 工具 schema 中使用的 Agent 类型描述字符串。"""
    return "\n".join(
        f"- {name}: {cfg['description']}"
        for name, cfg in AGENT_TYPES.items()
    )


def get_tools_for_agent(agent_type: str, base_tools: list) -> list:
    """
    根据 Agent 类型过滤可用工具列表。

    '*' 表示继承全部 base_tools。
    否则按白名单只保留指定工具名。

    注意：子 Agent 不获得 Task 工具，防止无限递归派生。
    """
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")

    if allowed == "*":
        return base_tools  # 全量工具，但不含 Task 本身

    # 按工具名白名单过滤
    return [t for t in base_tools if t["name"] in allowed]


# =============================================================================
# TASK 工具定义
# =============================================================================

TASK_TOOL = {
    "name": "Task",
    "description": f"""Spawn a subagent for a focused subtask.

Subagents run in ISOLATED context - they don't see parent's history.
Use this to keep the main conversation clean.

Agent types:
{get_agent_descriptions()}

Example uses:
- Task(explore): "Find all files using the auth module"
- Task(plan): "Design a migration strategy for the database"
- Task(code): "Implement the user registration form"
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short task name (3-5 words) for progress display"  # 用于终端进度展示的简短标题
            },
            "prompt": {
                "type": "string",
                "description": "Detailed instructions for the subagent"             # 传给子 Agent 的完整指令
            },
            "agent_type": {
                "type": "string",
                "enum": list(AGENT_TYPES.keys()),                                   # 枚举合法类型，防止传入无效值
                "description": "Type of agent to spawn"
            },
        },
        "required": ["description", "prompt", "agent_type"],  # 三个参数均为必填
    },
}


# =============================================================================
# 子 AGENT 执行逻辑
# =============================================================================

def run_task(description: str, prompt: str, agent_type: str,
             client, model: str, workdir, base_tools: list, execute_tool) -> str:
    """
    以隔离上下文执行一个子 Agent 任务。

    四个关键设计点：
    1. 独立历史（ISOLATED HISTORY）  - 子 Agent 从空白历史启动，看不到父对话
    2. 过滤工具（FILTERED TOOLS）    - 依据 agent_type 权限裁剪可用工具
    3. 专属提示（AGENT-SPECIFIC）    - 每种类型有针对性的行为约束
    4. 仅返回摘要（SUMMARY ONLY）    - 父 Agent 只获得最终结论，不获得中间过程

    Args:
        description: 用于终端进度显示的简短名称
        prompt:      子 Agent 接收的详细任务指令
        agent_type:  AGENT_TYPES 中的 key
        client:      Anthropic 客户端实例
        model:       使用的模型名称
        workdir:     工作目录
        base_tools:  工具定义列表（未过滤）
        execute_tool: 执行工具的函数（传入自身以支持递归调用）

    Returns:
        子 Agent 的最终文本输出
    """
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"  # 类型不合法时快速失败

    config = AGENT_TYPES[agent_type]

    # 根据 agent_type 构造专属系统提示词
    sub_system = f"""You are a {agent_type} subagent at {workdir}.

{config["prompt"]}

Complete the task and return a clear, concise summary."""

    # 按权限过滤工具列表
    sub_tools = get_tools_for_agent(agent_type, base_tools)

    # ★ 核心：子 Agent 使用全新的消息历史，与父 Agent 完全隔离
    # 父对话中的所有上下文对子 Agent 不可见
    sub_messages = [{"role": "user", "content": prompt}]

    # 在终端打印任务开始信息
    print(f"  [{agent_type}] {description}")
    start = time.time()   # 记录开始时间，用于计算耗时
    tool_count = 0        # 累计工具调用次数，用于进度展示

    # 子 Agent 主循环（与主 Agent 逻辑相同，但静默运行）
    while True:
        response = client.messages.create(
            model=model,
            system=sub_system,
            messages=sub_messages,
            tools=sub_tools,
            max_tokens=8000,
        )

        # 若停止原因不是工具调用，说明任务已完成，退出循环
        if response.stop_reason != "tool_use":
            break

        # 提取本轮所有工具调用块
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        results = []

        for tc in tool_calls:
            tool_count += 1                              # 累加调用计数
            output = execute_tool(tc.name, tc.input)    # 实际执行工具
            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,      # 与对应调用 id 绑定，API 要求
                "content": output
            })

            # 用 \r 覆写同一行，实现不换行的滚动进度显示
            elapsed = time.time() - start
            sys.stdout.write(
                f"\r  [{agent_type}] {description} ... {tool_count} tools, {elapsed:.1f}s"
            )
            sys.stdout.flush()  # 立即刷新缓冲区，确保进度实时可见

        # 将本轮助手回复和工具结果追加到子 Agent 历史，维持其内部上下文
        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    # 任务完成后打印最终耗时统计（换行结束进度条）
    elapsed = time.time() - start
    sys.stdout.write(
        f"\r  [{agent_type}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n"
    )

    # 从最终响应中提取纯文本——这是父 Agent 唯一能看到的内容
    # 中间过程的所有工具调用细节对父 Agent 完全透明不可见
    for block in response.content:
        if hasattr(block, "text"):
            return block.text

    return "(subagent returned no text)"  # 防御性兜底：子 Agent 未返回任何文本时的默认值


# =============================================================================
# 接入示例
# =============================================================================

"""
# 在主 Agent 的 execute_tool 函数中添加如下分支：

def execute_tool(name: str, args: dict) -> str:
    if name == "Task":
        return run_task(
            description=args["description"],
            prompt=args["prompt"],
            agent_type=args["agent_type"],
            client=client,
            model=MODEL,
            workdir=WORKDIR,
            base_tools=BASE_TOOLS,
            execute_tool=execute_tool  # 传入自身，支持子 Agent 递归调用
        )
    # ... 其他工具处理 ...


# 在主 Agent 的 TOOLS 列表中追加 Task 工具：
TOOLS = BASE_TOOLS + [TASK_TOOL]
"""