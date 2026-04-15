#!/usr/bin/env python3
# Harness: team mailboxes -- multiple models, coordinated through files.
"""
s09_agent_teams.py - Agent Teams

Persistent named agents with file-based JSONL inboxes. Each teammate runs
拥有基于文件的 JSONL 收件箱的、持久化且具名的智能体。
its own agent loop in a separate thread. Communication via append-only inboxes.
每个队友都在独立的线程中运行自己的智能体循环。通过“仅追加”的收件箱进行通信。

    Subagent (s04):  spawn -> execute -> return summary -> destroyed
                     生成 -> 执行 -> 返回总结 -> 销毁
    Teammate (s09):  spawn -> work -> idle -> work -> ... -> shutdown
                     生成 -> 工作 -> 空闲 -> 工作 -> ... -> 关机

    .team/config.json                   .team/inbox/
    +----------------------------+      +------------------+
    | {"team_name": "default",   |      | alice.jsonl      |
    |  "members": [              |      | bob.jsonl        |
    |    {"name":"alice",        |      | lead.jsonl       |
    |     "role":"coder",        |      +------------------+
    |     "status":"idle"}       |
    |  ]}                        |      send_message("alice", "fix bug"):
    +----------------------------+        open("alice.jsonl", "a").write(msg)

                                        read_inbox("alice"):
    spawn_teammate("alice","coder",...)   msgs = [json.loads(l) for l in ...]
         |                                open("alice.jsonl", "w").close()
         v                                return msgs  # drain
    Thread: alice             Thread: bob
    +------------------+      +------------------+
    | agent_loop       |      | agent_loop       |
    | status: working  |      | status: idle     |
    | ... runs tools   |      | ... waits ...    |
    | status -> idle   |      |                  |
    +------------------+      +------------------+

    5 message types (all declared, not all handled here):
    +-------------------------+-----------------------------------+
    | message                 | Normal text message               |
    | broadcast               | Sent to all teammates             |
    | shutdown_request        | Request graceful shutdown (s10)   |
    | shutdown_response       | Approve/reject shutdown (s10)     |
    | plan_approval_response  | Approve/reject plan (s10)         |
    +-------------------------+-----------------------------------+

Key insight: "Teammates that can talk to each other."
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path
import threading
import uuid

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TASKS_DIR = WORKDIR / ".tasks"
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

THRESHOLD = 50000
KEEP_RECENT = 3

# %% ------------ MessageBus and TeammateManager ------------

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# -- MessageBus: JSONL inbox per teammate --
# 消息总线：为每个“队友”提供一个基于JSONL文件的收件箱
class MessageBus:
    def __init__(self, inbox_dir: Path):
        # 收件箱目录（每个成员一个 .jsonl 文件）
        self.dir = inbox_dir
        # 创建目录（如果不存在）
        self.dir.mkdir(parents=True, exist_ok=True)

    # TODO: leader和sub都可以调用的工具，用于所有智能体间通信
    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """
        msg_type: str = "message"
        在智能体通信中，绝大多数情况（90%）都是普通的“聊天消息”。
        如果每次发消息都要写 msg_type="message" 会很啰嗦。
        当需要发送特殊指令（比如 broadcast，或者 shutdown_request）时，才需要指定。

        # 情况 A：不传 msg_type，默认就是 "message"
          BUS.send("alice", "bob", "你好")
          BUS.send("alice", "bob", "你好", msg_type="message")

        # 情况 B：显式指定 msg_type
        BUS.send("lead", "alice", "所有人停止工作", msg_type="shutdown_request")
        """

        """
        extra: dict 是一个“扩展槽”。
        标准的消息体只需要 from, content, timestamp。
        但有时候可能需要传递一些非标准的额外数据（比如任务的优先级、关联的文件名等）。
        把它设为可选，是为了保持函数的灵活性。如果没有额外数据，就不传，保持消息体干净。

        情况 A：不需要额外数据
          BUS.send("alice", "bob", "代码写完了")
          生成的 JSON: {"type": "message", "from": "alice", "content": "...", "timestamp": ...}

        情况 B：需要带额外数据
          BUS.send("alice", "bob", "任务完成", extra={"priority": "high", "file": "main.py"})
          生成的 JSON: {"type": "message", ..., "priority": "high", "file": "main.py"}
        """

        # 校验消息类型是否合法
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        # 构造消息体
        msg = {
            "type": msg_type,         # 消息类型（message / broadcast 等）
            "from": sender,           # 发送者
            "content": content,       # 消息内容
            "timestamp": time.time(),# 时间戳
        }

        # 如果有额外字段，合并进去
        if extra:
            msg.update(extra)

        # 收件箱路径（每个用户一个文件）
        inbox_path = self.dir / f"{to}.jsonl"

        # 以追加方式写入一行JSON（JSONL格式）
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        return f"Sent {msg_type} to {to}"
    
    # TODO: leader和sub都可以调用的工具
    # 读取磁盘上属于该智能体的专属文件，读取之后，它会立即清空文件内容
    def read_inbox(self, name: str) -> list:        
        # 获取指定用户的收件箱文件路径
        inbox_path = self.dir / f"{name}.jsonl"

        # 如果文件不存在，说明没有消息
        if not inbox_path.exists():
            return []

        messages = []

        # 逐行读取JSONL文件
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))  # 反序列化为字典

        # 读取后清空收件箱（“消费”模式）
        inbox_path.write_text("")

        return messages

    # 供leader使用，向所有队友广播（除了自己）
    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0

        for name in teammates:
            if name != sender:
                # 调用send发送广播消息
                self.send(sender, name, content, "broadcast")
                count += 1

        return f"Broadcast to {count} teammates"


# 全局消息总线实例
BUS = MessageBus(INBOX_DIR)


# -- TeammateManager: persistent named agents with config.json --
# 队友管理器：管理多个“智能体”，并持久化到配置文件
class TeammateManager:
    def __init__(self, team_dir: Path):
        # 团队目录
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)

        # 配置文件路径
        self.config_path = self.dir / "config.json"

        # 加载配置（成员信息等）
        self.config = self._load_config()

        # 存储运行中的线程：name -> thread
        self.threads = {}

    def _load_config(self) -> dict:
        # 如果配置文件存在，读取并解析
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())

        # 否则返回默认配置
        return {"team_name": "default", "members": []}

    def _save_config(self):
        # 将当前配置写回文件（带缩进）
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        # 根据名字查找成员
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    # 新建或征用一个子智能体，修改名单中的状态并运行它的loop
    def spawn(self, name: str, role: str, prompt: str) -> str:
        # 查找是否已有该成员
        member = self._find_member(name)

        if member:
            # 如果成员正在运行，不允许重复启动
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"

            # 更新状态和角色
            member["status"] = "working"
            member["role"] = role
        else:
            # 创建新成员
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)

        # 保存配置
        self._save_config()

        # 创建线程运行该智能体
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )

        # 保存线程引用
        self.threads[name] = thread

        # 启动线程
        thread.start()

        return f"Spawned '{name}' (role: {role})"

    # 子智能体的loop
    def _teammate_loop(self, name: str, role: str, prompt: str):
        """
        在 s04 的架构中，子智能体就像一个研究员。
            任务：父智能体派它去查资料。
            过程：它查了很多文件，思考了很多。
            输出：最后，它把所有发现总结成一段文字，返回给父智能体。
            代码体现：run_subagent() 函数会 return 一个字符串摘要。

        在 s09 的架构中，子智能体更像一个建筑工人。
            任务：Lead 派它去“写一个登录页面”。
            过程：它调用 write_file 工具，创建了 login.html。
            输出：它的“输出”就是那个新创建的 login.html 文件！
            代码体现：当它完成任务，不再调用工具时，循环 break，线程结束。没有 return 任何文字。
        """

        # 系统提示词（定义智能体身份）
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        # 初始对话消息
        messages = [{"role": "user", "content": prompt}]
        # 可用工具列表
        tools = self._teammate_tools()

        # 最多执行50轮（防止无限循环）
        for _ in range(50):
            # 读取收件箱消息
            inbox = BUS.read_inbox(name)
            # 将消息加入上下文
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})
            try:
                # 调用模型生成回复
                response = client.messages.create(
                    model=MODEL,system=sys_prompt,messages=messages,
                    tools=tools,max_tokens=8000,)
            except Exception:
                # 出错直接退出循环
                break
            # 将模型输出加入对话
            messages.append({"role": "assistant", "content": response.content})
            # 如果不是工具调用，则任务结束
            if response.stop_reason != "tool_use":
                break
            results = []
            # 处理模型发起的工具调用
            for block in response.content:
                if block.type == "tool_use":
                    # 执行工具
                    output = self._exec(name, block.name, block.input)
                    # 打印日志（调试用）
                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    # 构造工具返回结果
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })
            # 将工具结果作为用户消息喂回模型（继续对话）
            messages.append({"role": "user", "content": results})

        # 任务结束后更新成员状态
        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()

    # 子智能体的Handler
    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # 执行不同工具（调度层）

        if tool_name == "bash":
            return _run_bash(args["command"])

        if tool_name == "read_file":
            return _run_read(args["path"])

        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])

        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])

        if tool_name == "send_message":
            return BUS.send(
                sender,
                args["to"],
                args["content"],
                args.get("msg_type", "message")
            )

        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)

        # 未知工具
        return f"Unknown tool: {tool_name}"

    # 子智能体的工具定义
    def _teammate_tools(self) -> list:
        # 定义模型可用的工具（函数调用 schema）
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},

            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},

            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},

            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},

            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},

            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
        ]

    # 列出所有成员状态
    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    # 返回所有成员名字列表，主智能体调用 broadcast 时作为参数
    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


# 全局团队管理器实例
TEAM = TeammateManager(TEAM_DIR)


# %% -- TaskManager: CRUD with dependency graph, persisted as JSON files --
class TaskManager:
    # TaskManager 类：用于管理任务的完整工具箱
    # 任务以 JSON 文件形式存储在磁盘上，每个任务一个文件

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        # 把传入的目录路径保存为实例属性，之后所有方法都通过 self.dir 访问它
        self.dir.mkdir(exist_ok=True)
        # 创建该目录。exist_ok=True 表示"如果目录已存在就不报错，直接跳过"
        self._next_id = self._max_id() + 1
        # 调用 _max_id() 找出当前已有任务中最大的 ID，再加 1，作为下一个新任务的 ID
        # 比如已有 task_1.json、task_3.json，则 _max_id()=3，_next_id=4
        # 下划线前缀（_next_id）是 Python 约定，表示"内部使用，外部别随意访问"



    def _max_id(self) -> int:
        # 扫描任务目录，找出已有任务文件中最大的任务 ID

        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        # 这是一个列表推导式，等价于一个 for 循环，逐步拆解：
        #   self.dir.glob("task_*.json")
        #     → 找出目录下所有文件名匹配 "task_*.json" 的文件，* 是通配符
        #     → 例如找到：task_1.json, task_3.json, task_5.json
        #   f.stem
        #     → 取文件名（不含扩展名），如 "task_3.json" 的 stem 是 "task_3"
        #   .split("_")[1]
        #     → 按下划线切割字符串，"task_3" → ["task", "3"]，取索引1即 "3"
        #   int(...)
        #     → 把字符串 "3" 转成整数 3
        # 最终 ids 是所有任务 ID 的整数列表，如 [1, 3, 5]
        return max(ids) if ids else 0
        # 如果 ids 不为空，返回最大值；如果目录里没有任务文件，ids 是空列表，返回 0



    def _load(self, task_id: int) -> dict:
        # 根据任务 ID 从磁盘读取对应的 JSON 文件，返回字典

        path = self.dir / f"task_{task_id}.json"
        # Path 对象支持用 / 拼接路径（比字符串拼接更安全跨平台）
        # f"task_{task_id}.json" 是 f-string，会把变量 task_id 插入字符串
        # 例如 task_id=3 → path 是 "tasks/task_3.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        # 检查文件是否存在，如果不存在就抛出 ValueError 异常，终止程序并提示错误信息
        return json.loads(path.read_text())
        # path.read_text() → 把文件内容读取为字符串
        # json.loads(...)  → 把 JSON 格式的字符串解析成 Python 字典



    def _save(self, task: dict):
        # 把一个任务字典写入磁盘对应的 JSON 文件

        path = self.dir / f"task_{task['id']}.json"
        # 根据任务字典里的 id 字段构造文件路径
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))
        # json.dumps(task, ...) → 把 Python 字典转成 JSON 格式的字符串
        #   indent=2          → 格式化输出，每层缩进 2 个空格，方便人类阅读
        #   ensure_ascii=False → 允许中文等非 ASCII 字符直接写入，不转成 \uXXXX 转义序列
        # path.write_text(...) → 把字符串写入文件（文件不存在会创建，已存在会覆盖）



    def create(self, subject: str, description: str = "") -> str:
        # 创建一个新任务并保存，返回该任务的 JSON 字符串
        # subject: str      → 任务标题，必填
        # description: str = "" → 任务描述，选填，默认为空字符串

        task = {
            "id": self._next_id,         # 分配当前可用的 ID
            "subject": subject,           # 任务标题
            "description": description,   # 任务描述
            "status": "pending",          # 初始状态固定为 "pending"（待处理）
            "blockedBy": [],              # 阻塞依赖列表，初始为空（没有前置任务）
            "owner": "",                  # 负责人，初始为空
        } # 构造一个新任务字典，包含所有字段的初始值
        self._save(task)
        # 把新任务写入磁盘
        self._next_id += 1
        # ID 计数器加 1，确保下次创建的任务拿到不同的 ID
        return json.dumps(task, indent=2, ensure_ascii=False)
        # 返回格式化的 JSON 字符串，方便调用者查看或展示



    def get(self, task_id: int) -> str:
        # 根据 ID 查询单个任务，返回格式化 JSON 字符串

        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)
        # _load() 从磁盘读取任务字典，json.dumps() 再转回格式化字符串返回
        # 看起来多此一举，但统一了返回格式，外部调用者总是拿到字符串



    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, remove_blocked_by: list = None) -> str:
        # 更新一个已有任务的状态或依赖关系. 所有参数都有默认值 None，表示"不传就不修改该字段"

        task = self._load(task_id)
        # 先从磁盘读取现有数据(以字典形式)，避免覆盖未修改的字段
        if status:
            # 只有调用者传入了 status 参数才执行这段（None 和空字符串都视为"没传"）
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}") # 防止非法数据写入
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id) # 如果任务完成了，从其他任务的依赖列表中移除它
        if add_blocked_by:
            # 如果传入了需要新增的依赖 ID 列表
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
            # task["blockedBy"] + add_blocked_by → 合并两个列表
            # set(...) → 转成集合，自动去重（防止同一个依赖被加两次）
            # list(...) → 再转回列表，因为 JSON 不支持集合类型
        if remove_blocked_by:
            # 如果传入了需要移除的依赖 ID 列表
            task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]
        self._save(task) # 把修改后的任务字典写回磁盘
        return json.dumps(task, indent=2, ensure_ascii=False) # 返回更新后的任务 JSON 字符串



    def _clear_dependency(self, completed_id: int):
        """Remove completed_id from all other tasks' blockedBy lists."""
        # 当一个任务完成时，自动把它从所有其他任务的"被阻塞"列表中删除

        for f in self.dir.glob("task_*.json"):
            # 遍历目录中的每一个任务文件
            task = json.loads(f.read_text())
            # 把每个文件读取并解析成字典
            if completed_id in task.get("blockedBy", []):
                # task.get("blockedBy", []) → 安全地获取 blockedBy 字段
                task["blockedBy"].remove(completed_id)
                self._save(task) # 把修改保存回磁盘



    def list_all(self) -> str:
        # 列出所有任务，返回格式化的文本摘要

        tasks = []
        # 初始化一个空列表，用来收集所有任务字典
        files = sorted(self.dir.glob("task_*.json"),
            key=lambda f: int(f.stem.split("_")[1]))
        # 获取所有任务文件，并按任务 ID 从小到大排序
        # sorted(..., key=...) → 按 key 函数的返回值排序
        # lambda f: int(f.stem.split("_")[1])
        #   → 匿名函数：对每个文件 f，提取其 ID 数字作为排序依据
        #   → 确保输出顺序是 task_1, task_2, task_3...而非文件系统的随机顺序
        # sorted() 的第一个参数是 self.dir.glob("task_*.json")，它产出的每一个元素都是 Path 文件对象。
        # sorted() 在内部遍历这些元素时，会把每一个元素依次传给 key 函数，所以 f 自然就是 Path 文件对象。
        for f in files:
            tasks.append(json.loads(f.read_text()))
        # 依次读取每个文件，解析为字典，追加到 tasks 列表
        if not tasks:
            return "No tasks."
        # 如果没有任何任务文件，直接返回提示字符串
        lines = []
        # 初始化空列表，每个元素是一行文本
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            # 用字典映射把状态转成可视化符号：
            #   "pending"     → "[ ]"  （待处理，空框）
            #   "in_progress" → "[>]"  （进行中，箭头）
            #   "completed"   → "[x]"  （已完成，打叉）
            # .get(t["status"], "[?]") → 如果状态不在字典中，返回 "[?]" 作为默认值
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            # 如果任务有依赖（blockedBy 不为空列表），生成 " (blocked by: [1, 2])" 这样的附加信息
            # t.get("blockedBy") → 空列表 [] 在 Python 中是假值，所以没有依赖时 else 分支返回空字符串
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
            # 拼成一行，例如：
            #   "[ ] #1: 写周报"
            #   "[>] #2: 做 PPT (blocked by: [1])"
            #   "[x] #3: 开会"
        return "\n".join(lines)
        # 把所有行用换行符连接成一个字符串并返回
        # "\n".join(列表) 是 Python 中把列表拼成多行文本的标准写法


TASKS = TaskManager(TASKS_DIR)

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

    # # 取出摘要文本（第一个 text block）
    # summary = response.content[0].text
    # 某些模型文本块可能不是第一个元素；这里从“硬编码索引”改为“动态搜索”，增加了空值处理，防御性编程
    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    if not summary:
        summary = "No summary generated."

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



# %% -- BackgroundManager: threaded execution + notification queue --
# 后台任务管理器：用于在后台线程执行命令，并通过队列返回完成通知
class BackgroundManager:
    def __init__(self):
        # 存储所有任务：task_id -> {status, result, command}
        self.tasks = {}
        # 用于存储已完成任务的通知队列（供外部轮询获取）
        self._notification_queue = []
        # 线程锁，保证多线程访问队列时安全
        self._lock = threading.Lock()


    def run(self, command: str) -> str:
        """Start a background thread, return task_id immediately. 
           启动后台线程，返回task_id"""
        # 生成一个短任务ID（UUID前8位）
        task_id = str(uuid.uuid4())[:8]
        # 初始化任务状态为 running
        self.tasks[task_id] = {
            "status": "running",
            "result": None,
            "command": command
        }
        # 创建后台线程执行命令
        thread = threading.Thread(
            target=self._execute,          # 线程执行函数
            args=(task_id, command),       # 传入任务ID和命令
            daemon=True                   # 守护线程（主程序退出时自动结束）
        )
        # 启动线程
        thread.start()
        # 立即返回任务启动信息（不会阻塞）
        return f"Background task {task_id} started: {command[:80]}"


    def _execute(self, task_id: str, command: str):
        """Thread target: run subprocess, capture output, push to queue.
           线程目标:运行子流程，捕获输出，推入队列。"""
        try:
            # 执行系统命令
            r = subprocess.run(
                command,
                shell=True,              # 使用shell执行（支持字符串命令）
                cwd=WORKDIR,             # 指定工作目录
                capture_output=True,     # 捕获stdout和stderr
                text=True,               # 输出以字符串形式返回
                timeout=300              # 超时时间300秒
            )

            # 拼接标准输出和错误输出，并限制最大长度
            output = (r.stdout + r.stderr).strip()[:50000]

            # 标记任务完成
            status = "completed"

        except subprocess.TimeoutExpired:
            # 超时异常处理
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            # 其他异常处理
            output = f"Error: {e}"
            status = "error"

        # 更新任务状态和结果
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"

        # 将任务完成信息加入通知队列（线程安全）
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],              # 截断命令用于展示
                "result": (output or "(no output)")[:500],  # 截断结果用于通知
            })

    def check(self, task_id: str = None) -> str:
        """Check status of one task or list all."""
        if task_id:
            # 查询指定任务
            t = self.tasks.get(task_id)

            # 如果任务不存在
            if not t:
                return f"Error: Unknown task {task_id}"

            # 返回任务状态和结果（如果还在运行则显示running）
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"

        # 如果未指定task_id，则列出所有任务
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")

        # 返回所有任务列表
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications.
           在LLM调用之前，清空后台通知并作为系统消息注入"""
        # 加锁确保线程安全
        with self._lock:
            # 拷贝当前通知队列
            notifs = list(self._notification_queue)
            # 清空队列（避免重复读取）
            self._notification_queue.clear()

        # 返回通知列表
        return notifs


# 创建一个全局后台任务管理器实例
BG = BackgroundManager()



# %% ------------ Tool implementations(sandbox) ------------

# 安全地解析路径, 防止越界访问
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR): 
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def _run_bash(command: str) -> str:
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
def _run_read(path: str, limit: int = None) -> str:
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
def _run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


# 安全地编辑文件内容(限制路径)
def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# %% ----------- TOOOLS, ToolMap and System prompt -----------

# 工具列表(定义工具的名称、描述和输入格式)。模型会根据这个列表来决定调用哪个工具, 以及如何构造输入参数。
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the current workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
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
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "removeBlockedBy": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {"name": "background_run", "description": "Run command in background thread. Returns task_id immediately.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate that runs in its own thread.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates with name, role, status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate's inbox.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
]


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw_bash: _run_bash(kw_bash["command"]),
    "read_file":  lambda **pkg_read: _run_read(pkg_read["path"], pkg_read.get("limit")),
    "write_file": lambda **data_in: _run_write(data_in["path"], data_in["content"]),
    "edit_file":  lambda **kw_edit: _run_edit(kw_edit["path"], kw_edit["old_text"], kw_edit["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":    lambda **kw: "Manual compression requested.",
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("removeBlockedBy")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
    "background_run":   lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
        "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}

# 系统提示词
SYSTEM = f"""
You are a coding agent and a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes.
Planning: Use the todo tool to plan multi-step tasks (mark as in_progress when starting, completed when done). 
          Use the task tool to delegate exploration or subtasks. 
          Use background_run for long-running commands.
Knowledge: Use load_skill to access specialized knowledge for unfamiliar topics. Available skills: {SKILL_LOADER.get_descriptions()}.
Behavior: Prefer using tools over generating prose.
"""
# SUBAGENT_SYSTEM = f"""You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."""

# %% ---------------- Agent loop with nag reminder injection ----------------

def agent_loop(messages: list):
    rounds_since_todo = 0 #  in_process 计数器
    while True:

        # TODO:<<<<<<<<<<<<<<<< compact ----------------
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        # TODO:---------------- compact >>>>>>>>>>>>>>>>

        # TODO:<<<<<<<<<<<<<<<< background_task ----------------
        # Drain background notifications and inject as system message before LLM call
        notifs = BG.drain_notifications()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
        # TODO:---------------- background_task >>>>>>>>>>>>>>>>

        # TODO:<<<<<<<<<<<< agent_teams:read_inbox ------------
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
        # TODO:------------ agent_teams:read_inbox >>>>>>>>>>>>

        # 调用 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant","content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []

        # TODO:<<<<<<<<<<<< todo and compact ------------
        used_todo =False 
        manual_compact = False
        # TODO:------------ todo and compact >>>>>>>>>>>>

        for block in response.content:
            # 如果这个 block 是工具调用
            if block.type == "tool_use":

                # TODO:<<<<<<<<<<<< compact ------------
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
                # TODO:------------ compact >>>>>>>>>>>>

                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        # **block.input: AI 提供的参数被解包传入handler。如果工具名不存在(else), 返回错误字符串。
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"

                # 打印命令与部分输出
                print(f"> {block.name}:")
                print(str(output)[:200])
                # 收集结果
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # 对应 tool_use 的 ID
                    "content": output
                })
        # TODO:<<<<<<<<<<<<<<<<<<<<<< todo ----------------------
                if block.name == "todo":
                    used_todo = True
        # 模型连续 3 轮以上不调用 todo 时注入提醒。
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        # TODO:---------------------- todo >>>>>>>>>>>>>>>>>>>>>>>

        # 把工具执行结果作为“用户消息”喂回模型
        messages.append({"role": "user","content": results})

        # TODO:<<<<<<<<<<<<<<<<<<<<<< compact ---------------------
        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
        # TODO:-------------------- compact >>>>>>>>>>>>>>>>>>>>>>>



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

        # TODO:<<<< agent_teams:Command-line Debugger ----
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        # TODO:---- agent_teams:Command-line Debugger >>>>

        history.append({"role": "user","content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list): # 如果是结构化 block(列表)
            for block in response_content:
                if hasattr(block, "text"): # 有 text 属性就打印
                    print(block.text)
        print()  # 空行分隔


"""
采用 `文件即服务` 的设计思路，利用 JSONL 文件作为“邮箱”来实现进程间通信。

**生命周期管理**
    智能体不再是即用即弃，而是拥有状态流转：spawn (生成) -> WORKING (工作中) -> IDLE (空闲) -> SHUTDOWN (关闭)。
    TeammateManager：负责维护 config.json（团队名册），管理队友的生成和状态。

**通信机制：MessageBus**
    基于 .team/inbox/ 目录下的 JSONL 文件。
    发送 (send)：以“追加模式”向目标队友的 .jsonl 文件写入一行 JSON 数据。
    接收 (read_inbox)：读取目标文件的所有内容，解析为 JSON 数组，并清空文件（Drain-on-read）。

**运行逻辑**
    主程序 启动领导智能体。
    领导 可以调用工具 spawn 创建新线程运行队友智能体。
    队友 在循环中运行，每次调用 LLM 前会检查自己的收件箱。
    如果有新消息，将消息内容注入到 LLM 的上下文窗口中，队友据此做出反应。
"""