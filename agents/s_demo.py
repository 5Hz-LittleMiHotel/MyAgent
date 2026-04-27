#!/usr/bin/env python3
# Harness: directory isolation -- parallel execution lanes that never collide.
"""
s12_worktree_task_isolation.py - Worktree + Task Isolation
                                   工作树 + 任务隔离
Directory-level isolation for parallel task execution.
并行任务执行的目录级隔离。
Tasks are the control plane and worktrees are the execution plane.
任务是控制平面，工作树是执行平面。

    一个任务一个文件（One-file-per-task）：
    .tasks/task_12.json
      {
        "id": 12,
        "subject": "Implement auth refactor",
        "status": "in_progress",
        "worktree": "auth-refactor"
      }
      id, subject, status, owner: 业务状态（我们要做什么？做到哪了？谁负责？）。
      worktree: "auth-refactor" （关键点！）：注意，这里存的是 name（名字），而不是 path（路径）。
      这是一种极好的解耦。任务不需要知道 worktree 藏在 .worktrees/ 下面多深，也不需要知道 Git 分支叫什么。
      它只记住：“我绑定了一个叫 auth-refactor 的执行环境。”

    单一索引文件：
    .worktrees/index.json
      {
        "worktrees": [
          {
            "name": "auth-refactor",
            "path": ".../.worktrees/auth-refactor",
            "branch": "wt/auth-refactor",
            "task_id": 12,
            "status": "active"
          }
        ]
      }
      name, path, branch: 这是纯粹的物理实现细节（在哪？基于什么分支拉出来的？）。
      task_id: 12 （关键点！）：这是反向指针，指回控制面。

两个文件双向绑定。
崩溃后从 .tasks/ + .worktrees/index.json 重建现场。会话记忆是易失的; 磁盘状态是持久的。
假设 Agent 跑到一半，Python 进程被 kill 了，内存里的 TASKS 和 WORKTREES 对象全没了。
等下次重启，代码里的这两行会瞬间复活一切：
    TASKS = TaskManager(REPO_ROOT / ".tasks")                    # 扫描 task_*.json
    EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl") # 追加日志
    # WorktreeManager 初始化时读 index.json

Key insight: "Isolate by directory, coordinate by task ID."
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
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60


def detect_repo_root(cwd: Path) -> Path | None:
    """Return git repo root if cwd is inside a repo, else None.
       如果cwd在repo中，则返回git repo根，否则返回 None。         """
    try:
        # 调用 git 命令获取当前目录所属的仓库根路径
        # rev-parse --show-toplevel 是 Git 提供的标准获取根目录的命令
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,               # 以传入的当前工作目录作为基准去执行
            capture_output=True,   # 捕获标准输出和标准错误，不打印到终端
            text=True,             # 将输出的字节流自动解码为字符串
            timeout=10,            # 设置 10 秒超时，防止 git 进程意外挂死
        )
        # 检查退出码：非 0 通常意味着当前目录不在任何 git 仓库内
        if r.returncode != 0:
            return None
        # 取出输出结果，strip() 去掉末尾的换行符，并转为 Path 对象
        root = Path(r.stdout.strip())
        # 防御性检查：确保这个路径在磁盘上真实存在，存在才返回
        return root if root.exists() else None
    except Exception:
        # 兜底处理：如果系统没装 git (抛出 FileNotFoundError) 或权限不足等任何异常
        # 统一静默捕获，返回 None，保证 Agent 启动不会因为环境问题直接崩溃
        return None

REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR


# %% ------------ MessageBus and TeammateManager ------------

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_request",
    "plan_approval_response",
}

# -- Request trackers: correlate by request_id --
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock() # 请求追踪锁,保护 shutdown_requests、plan_requests
_claim_lock = threading.Lock() # 任务认领锁,保护任务文件系统 (.tasks/task_*.json) 以及 claim_task 函数中的逻辑

# -- MessageBus: JSONL inbox per teammate --
# 消息总线：为每个“队友”提供一个基于JSONL文件的收件箱
class MessageBus:
    def __init__(self, inbox_dir: Path):
        # 收件箱目录（每个成员一个 .jsonl 文件）
        self.dir = inbox_dir
        # FIX: MessageBus inbox drain lock
        # send() 和 read_inbox() 必须共用这把锁；否则消息可能刚好
        # 追加在 read_text() 和 write_text("") 之间，然后被清空覆盖。
        self._inbox_lock = threading.Lock()
        # 创建目录（如果不存在）
        self.dir.mkdir(parents=True, exist_ok=True)

    # TODO: leader和sub都可以调用的工具，用于所有智能体间通信
    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        # 校验消息类型是否合法
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        # 构造消息体
        msg = {
            "type": msg_type,         # 消息类型（message / broadcast 等）
            "from": sender,           # 发送者
            "content": content,       # 消息内容
            "timestamp": time.time(), # 时间戳
        }
        # 如果有额外字段，合并进去
        if extra:
            msg.update(extra)
        # 收件箱路径（每个用户一个文件）
        inbox_path = self.dir / f"{to}.jsonl"
        # 以追加方式写入一行JSON（JSONL格式）
        with self._inbox_lock:
            with open(inbox_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"Sent {msg_type} to {to}"
    

    # TODO: leader和sub都可以调用的工具
    # 读取磁盘上属于该智能体的专属文件，读取之后，它会立即清空文件内容
    def read_inbox(self, name: str) -> list:        
        # 获取指定用户的收件箱文件路径
        inbox_path = self.dir / f"{name}.jsonl"
        # 如果文件不存在，说明没有消息
        if not inbox_path.exists():
            return []
        with self._inbox_lock:
            messages = []
            # FIX: MessageBus atomic in-process drain
            # 读取和清空必须和 send() 互斥，避免读后清空覆盖新追加消息。
            for line in inbox_path.read_text(encoding="utf-8").strip().splitlines():
                if line:
                    messages.append(json.loads(line))  # 反序列化为字典
            # 读取后清空收件箱（“消费”模式）
            inbox_path.write_text("", encoding="utf-8")
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

# -- Task board scanning --
def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        if task.get("owner"):
            existing_owner = task.get("owner") or "someone else"
            return f"Error: Task {task_id} has already been claimed by {existing_owner}"
        if task.get("status") != "pending":
            status = task.get("status")
            return f"Error: Task {task_id} cannot be claimed because its status is '{status}'"
        if task.get("blockedBy"):
            return f"Error: Task {task_id} is blocked by other task(s) and cannot be claimed yet"
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


# -- Identity re-injection after compression --
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


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

    def _set_status(self, name: str, status: str):
        # s09:
        # if member:
        #     if member["status"] != "shutdown":
        #         member["status"] = "idle"
        #         self._save_config()
        # s10: 
        # 如果是因为 should_exit = True 退出的，状态就记为 "shutdown"
        # 如果是别的原因（比如转了50轮没事干了，或者报错了）退出的，状态记为 "idle"
        # s11:
        # 把s10的内容封装成set_status，为了使得idle和shutdown复用
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def _set_plan_waiting(self, name: str, request_id: str, plan_text: str):
        # FIX: persistent plan approval state machine
        # Persist the approval gate so a teammate cannot keep executing after
        # submitting a plan, and so /team can show what it is waiting on.
        member = self._find_member(name)
        if member:
            member["status"] = "waiting_approval"
            member["pending_plan_request_id"] = request_id
            member["pending_plan"] = plan_text
            self._save_config()

    def _clear_plan_waiting(self, name: str, status: str = "idle"):
        # FIX: persistent plan approval state machine
        # Approval/rejection clears the pending fields; the teammate returns to
        # idle until its loop resumes actual work.
        member = self._find_member(name)
        if member:
            member["status"] = status
            member.pop("pending_plan_request_id", None)
            member.pop("pending_plan", None)
            self._save_config()

    def _is_waiting_approval(self, name: str) -> bool:
        member = self._find_member(name)
        return bool(member and member.get("status") == "waiting_approval")

    def _find_pending_plan_request(self, request_id: str) -> dict | None:
        # FIX: Recover pending plan approvals from persisted teammate config after restart.
        for member in self.config.get("members", []):
            if member.get("pending_plan_request_id") == request_id:
                return {
                    "from": member.get("name", ""),
                    "plan": member.get("pending_plan", ""),
                    "status": "pending",
                }
        return None

    def _handle_plan_approval_response(self, name: str, msg: dict) -> str:
        # FIX: teammate consumes plan approval replies
        # Treat approval responses as protocol messages, not ordinary chat.
        member = self._find_member(name)
        if not member:
            return json.dumps(msg)
        expected = member.get("pending_plan_request_id")
        req_id = msg.get("request_id", "")
        if expected and req_id != expected:
            return (
                f"<plan_approval_ignored request_id=\"{req_id}\" "
                f"expected=\"{expected}\">Mismatched approval response.</plan_approval_ignored>"
            )
        approve = bool(msg.get("approve"))
        feedback = msg.get("feedback", "")
        self._clear_plan_waiting(name, "idle")
        if approve:
            return (
                f"<plan_approved request_id=\"{req_id}\">"
                f"You may proceed with the approved plan.</plan_approved>"
            )
        return (
            f"<plan_rejected request_id=\"{req_id}\">"
            f"Feedback: {feedback}\nRevise your plan before doing major work."
            f"</plan_rejected>"
        )

    def _drain_protocol_inbox(self, name: str, messages: list) -> list:
        # FIX: teammate inbox protocol handling
        # Approval replies update the persisted gate before the LLM sees them.
        inbox = BUS.read_inbox(name)
        for msg in inbox:
            if msg.get("type") == "plan_approval_response":
                messages.append({
                    "role": "user",
                    "content": self._handle_plan_approval_response(name, msg),
                })
            else:
                messages.append({"role": "user", "content": json.dumps(msg)})
        return inbox

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
        # 保存线程引用,并启动线程
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    # 子智能体的loop
    def _teammate_loop(self, name: str, role: str, prompt: str):
        # TODO: 没做强制关机。这里的关机都是teammate决定的。
        team_name = self.config["team_name"]
        # 系统提示词（定义智能体身份）
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
        )
        # 初始对话消息
        messages = [{"role": "user", "content": prompt}]
        # 可用工具列表
        tools = self._teammate_tools()
        # 默认不退出，继续干活
        should_exit = False
        while True:
            # 最多执行50轮（防止无限循环）
            for _ in range(50):
                # FIX: teammate inbox protocol handling
                # Approval responses are consumed by the state machine before
                # ordinary messages are appended to model context.
                self._drain_protocol_inbox(name, messages)
                # 检查旗帜，真正的退出
                if should_exit:
                    self._set_status(name, "shutdown")
                    return
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
                idle_requested = False
                # 处理模型发起的工具调用
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "idle":
                            idle_requested = True
                            output = "Entering idle phase. Will poll for new tasks."
                        else:
                        # 执行工具
                            output = self._exec(name, block.name, block.input)
                        # 打印日志
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")
                        # 构造工具返回结果
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output),
                        })
                        # shutdown的核心逻辑：
                        # 当领导发来 shutdown_request，teammate 思考后调用了 shutdown_response 工具，并且参数 approve=true 后，
                        # 改变 should_exit 旗杆状态
                        if block.name == "shutdown_response" and block.input.get("approve"):
                            should_exit = True
                        if block.name == "plan_approval" and self._is_waiting_approval(name):
                            # FIX: Stop the work loop immediately after submitting a plan.
                            idle_requested = True
                # 将工具结果作为用户消息喂回模型（继续对话）
                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break
            
            # -- IDLE PHASE: poll for inbox messages and unclaimed tasks --
            if not self._is_waiting_approval(name):
                self._set_status(name, "idle") # 50轮结束自动进入idle
            resume = False # 初始化恢复标志：默认假设无任务不工作
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1) # 计算最大轮询次数
            for _ in range(polls): # 进入轮询循环
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name) # --- 检查点 A：有没有消息
                if inbox: # 如果有消息，遍历处理
                    for msg in inbox:
                        if msg.get("type") == "plan_approval_response":
                            # FIX: teammate consumes plan approval replies
                            messages.append({
                                "role": "user",
                                "content": self._handle_plan_approval_response(name, msg),
                            })
                            continue
                        # 特殊情况：如果是关机指令，强制退出
                        if msg.get("type") == "shutdown_request":
                            # FIX: idle shutdown acknowledgement
                            # Even while idle, keep the shutdown protocol consistent:
                            # update the lead-side tracker and send a shutdown_response
                            # before the teammate exits.
                            req_id = msg.get("request_id", "")
                            if req_id:
                                with _tracker_lock:
                                    if req_id in shutdown_requests:
                                        shutdown_requests[req_id]["status"] = "approved"
                            BUS.send(
                                name,
                                "lead",
                                "Idle shutdown acknowledged.",
                                "shutdown_response",
                                {"request_id": req_id, "approve": True},
                            )
                            self._set_status(name, "shutdown")
                            return
                        # 普通消息：将 JSON 格式的消息转为字符串，加入上下文
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    # 有消息处理，设置标志位并跳出轮询，回到 WORK 阶段
                    resume = True
                    break
                if self._is_waiting_approval(name):
                    # FIX: plan approval wait loop
                    # While approval is pending, do not auto-claim unrelated
                    # tasks and do not leave the waiting state.
                    continue
                # --- 检查点 B：有没有未认领的任务（sort过取第一个可用任务）
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    result = claim_task(task["id"], name) # 尝试认领任务（内部带锁）
                    if result.startswith("Error:"): # 如果认领失败进入下一次轮询循环，继续找任务
                        continue
                    # 认领成功，构造任务提示语，告诉 LLM 它刚刚自动认领了什么
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    # --- 关键：身份重注入 ---
                    # 如果上下文太短（len <= 3），说明可能发生了严重的上下文压缩或重置
                    # 此时 LLM 可能忘了自己是谁。因此必须重新插入身份块。
                    if len(messages) <= 3:
                        # 在最前面插入 teammate 的身份定义
                        messages.insert(0, make_identity_block(name, role, team_name))
                        # 紧接着插入一个助手的回复，模拟 LLM 说“我回来了”，让对话衔接更自然
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    # 将任务信息加入上下文
                    messages.append({"role": "user", "content": task_prompt})
                    # 模拟 LLM 已经接受了任务（预设一个 assistant 回复），引导 LLM 接着干活
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    # 找到活了，设置标志位并跳出轮询，回到 WORK 阶段
                    resume = True
                    break
            # --- 超时处理 ---
            # 如果 for 循环跑完了（12次都检查了），resume 还是 False，说明既没消息也没任务，等了60秒
            if not resume:
                if self._is_waiting_approval(name):
                    # FIX: plan approval wait loop
                    # Keep polling for lead approval instead of timing out to
                    # shutdown while the persisted gate is still active.
                    continue
                self._set_status(name, "shutdown")
                return
            # --- 恢复工作 ---
            # 如果 resume 是 True（说明上面 break 出来了），更新状态为 working
            # while True 的下一次循环会回到 WORK 阶段继续处理新的 messages
            self._set_status(name, "working")


    # 子智能体的Handler
    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # 执行不同工具（调度层）

        safe_while_waiting = {"read_inbox", "send_message", "idle", "task_get"}
        if self._is_waiting_approval(sender) and tool_name not in safe_while_waiting:
            # FIX: plan approval execution gate
            # Once a plan is submitted, block mutating/active tools until the
            # lead's plan_approval_response clears the persisted gate.
            member = self._find_member(sender) or {}
            req_id = member.get("pending_plan_request_id", "")
            return (
                "Error: waiting for plan approval"
                f"{f' (request_id={req_id})' if req_id else ''}; "
                "only read_inbox, send_message, idle, and task_get are allowed."
            )

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
            inbox = BUS.read_inbox(sender)
            processed = []
            for msg in inbox:
                if msg.get("type") == "plan_approval_response":
                    processed.append(self._handle_plan_approval_response(sender, msg))
                else:
                    processed.append(msg)
            return json.dumps(processed, indent=2, ensure_ascii=False)
        if tool_name == "task_get":
            return TASKS.get(args["task_id"])
        if tool_name == "shutdown_response":
        # ==========================================
        # 协议 1 处理：关机响应
        # 决策权在 Teammate 手中，由 Teammate 触发并修改状态
        # ==========================================
            # 从 LLM 传来的参数中提取 "挂号条" ID，用于在 Tracker 中定位这笔请求
            req_id = args["request_id"]
            # 提取 Teammate LLM 的最终决定：True 表示同意关机，False 表示拒绝关机
            approve = args["approve"]
            
            # 【加锁】因为多个 Teammate 线程可能同时修改全局字典，必须用锁保证数据安全
            with _tracker_lock:
                # 防御性编程：检查这个 req_id 是不是 Lead 真的发出来的，防止 LLM 幻觉乱造 ID
                if req_id in shutdown_requests:
                    # 【核心：驱动 FSM】根据 approve 布尔值，将状态从 pending 流转为 approved 或 rejected
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
            
            # 状态在本地落盘后，通过消息总线把结果回传给 Lead
            BUS.send(
                sender, "lead", 
                args.get("reason", ""), # 如果 Teammate 拒绝，这里会带上拒绝理由；没理由默认空字符串
                "shutdown_response",    # 消息类型标识
                {"request_id": req_id, "approve": approve}, # 结构化载荷：必须带上 req_id 给 Lead 对账
            )
            # 返回执行结果给当前 Teammate 的 LLM，让它知道自己的工具调用成功了，return后在teammate loop中触发 should_exit 逻辑
            return f"Shutdown {'approved' if approve else 'rejected'}"
        if tool_name == "plan_approval":
        # ==========================================
        # 协议 2 处理：计划提交
        # 决策权在 Lead 手中，Teammate 只负责发起请求并创建 pending 状态
        # ==========================================
            # 提取 Teammate 想要执行的计划文本内容
            plan_text = args.get("plan", "")
            # 【生成挂号条】由发起方生成唯一的 8 位 request_id
            req_id = str(uuid.uuid4())[:8]
            
            # 【加锁】保护共享字典的写入操作
            with _tracker_lock:
                # 在 Tracker 中为这个新计划开一个房间，初始状态强制设为 pending
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            self._set_plan_waiting(sender, req_id, plan_text)
            
            # 通过消息总线把计划内容和 req_id 打包发送给 Lead 进行审查
            BUS.send(
                sender, "lead", 
                plan_text,                   # 计划正文作为消息内容发给 Lead 看
                "plan_approval_request",     # FIX: clearer plan approval message naming
                {"request_id": req_id, "plan": plan_text}, # 结构化载荷
            )
            # 返回结果给 Teammate 的 LLM，告诉它计划已提交，必须停下来等待 Lead 的批复
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
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

            {"name": "idle", "description": "Enter idle polling while waiting for messages or tasks.",
             "input_schema": {"type": "object", "properties": {}}},

            {"name": "task_get", "description": "Read-only: get full details of a task by ID.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},

            {"name": "shutdown_response", "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},

            {"name": "plan_approval", "description": "Submit a plan for lead approval. Provide plan text.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},

            {"name": "claim_task", "description": "Claim a task from the task board by ID.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
        ]

    # 列出所有成员状态
    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            pending = ""
            if m.get("pending_plan_request_id"):
                # FIX: visible persistent plan approval state
                pending = f" waiting_for_plan={m['pending_plan_request_id']}"
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}{pending}")
        return "\n".join(lines)

    # 返回所有成员名字列表，主智能体调用 broadcast 时作为参数
    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


# 全局团队管理器实例
TEAM = TeammateManager(TEAM_DIR)


# %% -- EventBus: append-only lifecycle events for observability --
# 追加写入的生命周期事件总线，用于系统状态的可观测性。
# "append-only" 意味着只会往文件末尾加新内容，永远不会修改或删除历史内容。
class EventBus:
    def __init__(self, event_log_path: Path):
        # 将传入的事件日志文件路径（如 .worktrees/events.jsonl）保存为实例变量
        self.path = event_log_path
        # 确保日志文件的父目录（.worktrees/）存在。
        # parents=True 允许创建多级目录，exist_ok=True 表示如果目录已存在不报错。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 如果日志文件不存在，则创建一个空文件。
        # 这是为了防止后续使用 "a" 模式追加写入时，因文件不存在而报错。
        if not self.path.exists():
            self.path.write_text("")

    def emit(
        self,
        event: str,
        task: dict | None = None,
        worktree: dict | None = None,
        error: str | None = None,
    ):
        # 构建要写入文件的事件数据字典（负载）
        payload = {
            "event": event,             # 事件名称，如 "worktree.create.after"
            "ts": time.time(),          # 时间戳，记录事件发生的精确 Unix 时间（浮点数秒）
            "task": task or {},         # 关联的任务信息。如果传入了 task 字典则使用，否则给个空字典防 None
            "worktree": worktree or {}, # 关联的工作树信息。逻辑同上
        }
        # 如果调用时传入了 error 字符串，说明是异常事件，将其加入 payload
        if error:
            payload["error"] = error
        
        # 打开文件并写入。模式 "a" 代表 append（追加模式）。
        # 技术细节：追加模式在操作系统层面会将文件指针移到末尾，即使在多线程/多进程并发写入时，
        # 只要单次写入的字节数不超过系统管道缓冲区（通常几KB），就不会发生内容交叉覆盖。
        # encoding="utf-8" 确保中文等字符不会乱码。
        with self.path.open("a", encoding="utf-8") as f:
            # 将 payload 字典序列化为 JSON 字符串，并加上换行符 "\n"。
            # 这就是 JSONL（JSON Lines）格式：每行是一个独立的 JSON 对象，方便按行读取和解析。
            f.write(json.dumps(payload) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        # 边界值安全处理：将 limit 限制在 1 到 200 之间。
        # 防止外部传入负数、0 或极大的数字（如 100000）导致后续切片异常或内存溢出。
        n = max(1, min(int(limit or 20), 200))
        
        # 读取整个日志文件内容，并按换行符拆分成列表。
        # 技术局限：当日志文件极大时（如几百MB），这种全量读取会占用较多内存。
        # 但在 Agent 的单次会话场景下，日志量通常很小，这种实现最简单可靠。
        lines = self.path.read_text(encoding="utf-8").splitlines()
        
        # 使用 Python 切片特性 [-n:]，获取列表最后 n 个元素（即最近的 n 行日志）
        recent = lines[-n:]
        
        items = []
        # 遍历这几行日志字符串
        for line in recent:
            try:
                # 尝试将每行字符串反序列化为 Python 字典，加入结果列表
                items.append(json.loads(line))
            except Exception:
                # 防御性编程：如果某一行日志被意外截断或损坏（不是合法 JSON），
                # 不抛出异常中断程序，而是将其包装成一个特殊的错误字典，保留原始文本供排查。
                items.append({"event": "parse_error", "raw": line})
        
        # 将解析后的字典列表，格式化输出为带缩进的 JSON 字符串返回给 Agent 阅读
        return json.dumps(items, indent=2)

EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")

# %% -- TaskManager: CRUD with dependency graph, persisted as JSON files --
class TaskManager:
    # TaskManager 类：用于管理任务的完整工具箱
    # 任务以 JSON 文件形式存储在磁盘上，每个任务一个文件

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        # 把传入的目录路径保存为实例属性，之后所有方法都通过 self.dir 访问它
        self.dir.mkdir(parents=True, exist_ok=True)
        # 创建该目录。exist_ok=True 表示"如果目录已存在就不报错，直接跳过"
        self._next_id = self._max_id() + 1
        # 调用 _max_id() 找出当前已有任务中最大的 ID，再加 1，作为下一个新任务的 ID
        # 比如已有 task_1.json、task_3.json，则 _max_id()=3，_next_id=4
        # 下划线前缀（_next_id）是 Python 约定，表示"内部使用，外部别随意访问"



    def _max_id(self) -> int:
        # 扫描任务目录，找出已有任务文件中最大的任务 ID
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                pass
        return max(ids) if ids else 0



    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"
        # Path 对象支持用 / 拼接路径（比字符串拼接更安全跨平台）
        # f"task_{task_id}.json" 是 f-string，会把变量 task_id 插入字符串
        # 例如 task_id=3 → path 是 "tasks/task_3.json"



    def _load(self, task_id: int) -> dict:
        # 根据任务 ID 从磁盘读取对应的 JSON 文件，返回字典
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        # 检查文件是否存在，如果不存在就抛出 ValueError 异常，终止程序并提示错误信息
        return json.loads(path.read_text())
        # path.read_text() → 把文件内容读取为字符串
        # json.loads(...)  → 把 JSON 格式的字符串解析成 Python 字典



    def _save(self, task: dict):
        # 把一个任务字典写入磁盘对应的 JSON 文件
        path = self._path(task['id'])
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
            "worktree": "",
            "blockedBy": [],
            "created_at": time.time(),
            "updated_at": time.time(),
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


    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()


    def update(self, task_id: int, status: str = None, owner: str = None,
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
        if owner is not None:
            task["owner"] = owner
        task["updated_at"] = time.time()        
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



    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        """
        将任务与 Worktree 进行双向绑定。
        这是 s12 架构的核心：一旦绑定，任务状态自动流转为 'in_progress'。
        """
        # 加载任务数据：从 .tasks/task_<id>.json 读取当前任务信息
        task = self._load(task_id)
        
        # 建立关联：在任务记录中写入绑定的 worktree 名称
        # 这样任务就知道自己“在哪做”了
        task["worktree"] = worktree
        
        if owner:
            task["owner"] = owner

        # 状态机流转：
        # 如果任务当前是 'pending'（待办），绑定工作树意味着开始干活了
        # 所以自动将状态推进到 'in_progress'（进行中）
        if task.get("status") == "pending":
            task["status"] = "in_progress"

        # 更新时间戳：记录最后一次修改的时间，用于调试和排序
        task["updated_at"] = time.time()
            
        # 5. 持久化：将更新后的任务数据写回磁盘
        self._save(task)
        
        # 6. 返回结果：返回更新后的任务 JSON 字符串，供调用者确认
        return json.dumps(task, indent=2)

    def unbind_worktree(self, task_id: int) -> str:
        """
        解除任务与 Worktree 的绑定。
        通常在 Worktree 被删除（remove）时调用，用于清理任务状态。
        """
        # 1. 加载任务数据：读取当前任务信息
        task = self._load(task_id)
        
        # 2. 清除关联：将 worktree 字段置为空字符串
        # 表示该任务不再关联任何执行环境
        task["worktree"] = ""
        
        # 3. 更新时间戳
        task["updated_at"] = time.time()
        
        # 4. 持久化：保存更改到磁盘
        self._save(task)
        
        # 5. 返回结果：返回更新后的任务 JSON 字符串
        return json.dumps(task, indent=2)



    def list_all(self) -> str:
        # 列出所有任务，返回格式化的文本摘要
        tasks = []
        # 初始化一个空列表，用来收集所有任务字典
        files = sorted(self.dir.glob("task_*.json"), key=lambda f: int(f.stem.split("_")[1]))
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
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            wt = f" wt={t['worktree']}" if t.get("worktree") else ""
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            # 如果任务有依赖（blockedBy 不为空列表），生成 " (blocked by: [1, 2])" 这样的附加信息
            # t.get("blockedBy") → 空列表 [] 在 Python 中是假值，所以没有依赖时 else 分支返回空字符串
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{blocked}{wt}")
            # 拼成一行，例如：
            #   "[ ] #1: 写周报"
            #   "[>] #2: 做 PPT (blocked by: [1])"
            #   "[x] #3: 开会"
        return "\n".join(lines)
        # 把所有行用换行符连接成一个字符串并返回
        # "\n".join(列表) 是 Python 中把列表拼成多行文本的标准写法


TASKS = TaskManager(TASKS_DIR)


# %% -- WorktreeManager: create/list/run/remove git worktrees + lifecycle index --
class WorktreeManager:
    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        self.repo_root = repo_root # Git 仓库的根目录路径
        self.tasks = tasks       # 任务管理器实例，用于绑定任务状态
        self.events = events     # 事件总线实例，用于记录生命周期日志
        self.dir = repo_root / ".worktrees" # 定义 worktree 的存储目录（在仓库根下）
        self.dir.mkdir(parents=True, exist_ok=True) # 确保目录存在，不存在则创建
        self.index_path = self.dir / "index.json" # 索引文件路径，用于持久化记录所有 worktree
        # 初始化时检查当前目录是否为有效的 Git 仓库
        # 如果不是仓库，后续的 git worktree 命令将不可用
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        """私有方法：检查是否在 Git 仓库中"""
        try:
            # 调用 git 命令检查是否在工作树内
            # --is-inside-work-tree 如果在仓库内返回 0，否则报错
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root, # 在仓库根目录执行
                capture_output=True, # 捕获输出
                text=True, # 返回字符串
                timeout=10,
            )
            # 如果命令执行成功（returncode == 0），说明是 Git 仓库
            return r.returncode == 0
        except Exception:
            # 捕获所有异常（如未安装 Git），返回 False
            return False

    def _run_git(self, args: list[str]) -> str:
        """私有方法：在仓库根目录运行 Git 命令"""
        if not self.git_available:
            # 如果检测到不是 Git 仓库，直接报错，防止误操作
            raise RuntimeError("Not in a git repository. worktree tools require git.")
            
        # 执行 git 命令，传入 args 列表（例如 ["worktree", "add", ...]）
        r = subprocess.run(
            ["git", *args], # 拼接 git 和参数
            cwd=self.repo_root, # 强制在仓库根目录执行，防止 cwd 错乱
            capture_output=True,
            text=True,
            timeout=120, # 设置较长超时，因为 git 操作可能较慢
        )
        
        # 检查命令是否执行失败
        if r.returncode != 0:
            # 如果失败，拼接 stdout 和 stderr 作为错误信息
            msg = (r.stdout + r.stderr).strip()
            # 抛出运行时错误，包含具体的 Git 错误信息
            raise RuntimeError(msg or f"git {' '.join(args)} failed")
            
        # 命令成功，返回输出内容
        return (r.stdout + r.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        """私有方法：从磁盘加载索引文件"""
        # 读取 JSON 文件并解析为字典
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: dict):
        """私有方法：将数据保存回索引文件"""
        # 将字典写回 JSON 文件，格式化缩进以便人类阅读
        self.index_path.write_text(json.dumps(data, indent=2))

    def _find(self, name: str) -> dict | None:
        """私有方法：根据名称在索引中查找 worktree 记录"""
        idx = self._load_index() # 先加载索引
        # 遍历索引中的 worktrees 列表
        for wt in idx.get("worktrees", []):
            if wt.get("name") == name: # 找到匹配名称的条目
                return wt # 返回该条目
        return None # 未找到返回 None

    def _validate_name(self, name: str):
        """私有方法：校验 worktree 名称的合法性"""
        # 使用正则表达式限制名称格式：仅允许字母、数字、点、下划线、横线
        # 长度 1-40 字符，防止路径注入或非法文件名
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
            )

    def create(self, name: str, task_id: int = None, base_ref: str = "HEAD") -> str:
        """公开方法：创建一个新的 Git Worktree"""
        self._validate_name(name) # 1. 校验名称
        
        # 2. 检查索引中是否已存在同名 worktree
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")
            
        # 3. 检查任务 ID 是否存在（如果提供了 task_id）
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        # 4. 定义路径和分支名
        # 物理路径：.worktrees/<name>
        path = self.dir / name 
        # Git 分支名：wt/<name> (命名空间隔离)
        branch = f"wt/{name}" 

        # 5. 发布事件：创建开始
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else {},
            worktree={"name": name, "base_ref": base_ref},
        )
        
        try:
            # 6. 执行 Git 命令创建
            # git worktree add -b <branch> <path> <base_ref>
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])
            
            # 7. 构造索引条目
            entry = {
                "name": name,
                "path": str(path), # 记录绝对路径（相对于 repo root）
                "branch": branch,  # 记录对应的分支
                "task_id": task_id, # 绑定的任务 ID
                "status": "active", # 状态设为活跃
                "created_at": time.time(), # 记录时间戳
            }
            
            # 8. 更新索引文件
            idx = self._load_index()
            idx["worktrees"].append(entry) # 添加新条目
            self._save_index(idx) # 保存文件

            # 9. 如果绑定了任务，更新任务状态为 in_progress
            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)

            # 10. 发布事件：创建成功
            self.events.emit(
                "worktree.create.after",
                task={"id": task_id} if task_id is not None else {},
                worktree={ "name": name, "path": str(path), "branch": branch, "status": "active" },
            )
            
            # 11. 返回创建的条目信息（JSON 字符串）
            return json.dumps(entry, indent=2)
            
        except Exception as e:
            # 12. 异常处理：记录失败事件并重新抛出异常
            self.events.emit(
                "worktree.create.failed",
                task={"id": task_id} if task_id is not None else {},
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),
            )
            raise

    def list_all(self) -> str:
        """公开方法：列出所有受管理的 worktree"""
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "No worktrees in index."
            
        lines = []
        for wt in wts:
            # 拼接状态、名称、路径、分支和关联的任务 ID
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt['name']} -> "
                f"{wt['path']} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    def status(self, name: str) -> str:
        """公开方法：查看指定 worktree 的 Git 状态"""
        wt = self._find(name) # 1. 查找
        if not wt:
            return f"Error: Unknown worktree '{name}'"
            
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
            
        # 2. 在 worktree 目录下执行 git status
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (r.stdout + r.stderr).strip()
        return text or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        """公开方法：在指定 worktree 目录中运行 Shell 命令"""
        # 1. 安全检查：拦截危险命令（rm -rf / 等）
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"
            
        wt = self._find(name) # 2. 查找 worktree
        if not wt:
            return f"Error: Unknown worktree '{name}'"
            
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
            
        try:
            # 3. 执行命令
            # shell=True 允许执行复杂命令
            # cwd=path 确保命令在隔离的 worktree 目录中运行
            r = subprocess.run(
                command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=300, # 5分钟超时
            )
            out = (r.stdout + r.stderr).strip()
            # 4. 截断过长的输出，防止 Token 爆炸
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        """公开方法：移除 worktree（物理删除目录）"""
        wt = self._find(name) # 1. 查找
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        # 2. 发布事件：移除开始
        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )
        
        try:
            # 3. 构造 git worktree remove 命令参数
            args = ["worktree", "remove"]
            if force: # 是否强制删除（即使有未提交更改）
                args.append("--force")
            args.append(wt["path"]) # 指定要删除的路径
            
            self._run_git(args) # 执行删除

            # 4. 如果设置了 complete_task，同时更新关联的任务状态
            if complete_task and wt.get("task_id") is not None:
                task_id = wt["task_id"]
                before = json.loads(self.tasks.get(task_id)) # 获取旧状态用于日志
                self.tasks.update(task_id, status="completed") # 标记任务完成
                self.tasks.unbind_worktree(task_id) # 解绑 worktree
                
                # 发布任务完成事件
                self.events.emit(
                    "task.completed",
                    task={ "id": task_id, "subject": before.get("subject", ""), "status": "completed" },
                    worktree={"name": name},
                )

            # 5. 更新本地索引文件，标记状态为 removed
            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)

            # 6. 发布事件：移除成功
            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )
            return f"Removed worktree '{name}'"
            
        except Exception as e:
            # 7. 异常处理
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise

    def keep(self, name: str) -> str:
        """公开方法：标记 worktree 为保留状态（不自动删除）"""
        wt = self._find(name) # 1. 查找
        if not wt:
            return f"Error: Unknown worktree '{name}'"
            
        # 2. 更新索引中的状态为 'kept'
        idx = self._load_index()
        kept = None
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)

        # 3. 发布保留事件
        self.events.emit(
            "worktree.keep",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={ "name": name, "path": wt.get("path"), "status": "kept" },
        )
        
        # 4. 返回结果
        return json.dumps(kept, indent=2) if kept else f"Error: Unknown worktree '{name}'"

WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)


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
        lines.append(f"({done}/{len(self.items)} completed)")
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


# -- Lead-specific protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        # FIX: Rebuild missing in-memory plan request from persisted waiting state.
        req = TEAM._find_pending_plan_request(request_id)
        if not req:
            return f"Error: Unknown plan request_id '{request_id}'"
        with _tracker_lock:
            plan_requests[request_id] = req
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


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
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "owner": {"type": "string"}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "removeBlockedBy": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
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
    {"name": "shutdown_request", "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check the status of a shutdown request by request_id.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state (for lead -- rarely used).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {
        "name": "task_bind_worktree",
        "description": "Bind a task to a worktree name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "worktree": {"type": "string"},
                "owner": {"type": "string"},
            },
            "required": ["task_id", "worktree"],
        },
    },
    {
        "name": "worktree_create",
        "description": "Create a git worktree and optionally bind it to a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "task_id": {"type": "integer"},
                "base_ref": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_list",
        "description": "List worktrees tracked in .worktrees/index.json.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "worktree_status",
        "description": "Show git status for one worktree.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_run",
        "description": "Run a shell command in a named worktree directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "command": {"type": "string"},
            },
            "required": ["name", "command"],
        },
    },
    {
        "name": "worktree_remove",
        "description": "Remove a worktree and optionally mark its bound task completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "force": {"type": "boolean"},
                "complete_task": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_keep",
        "description": "Mark a worktree as kept in lifecycle state without removing it.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_events",
        "description": "List recent worktree/task lifecycle events from .worktrees/events.jsonl.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
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
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner"), kw.get("addBlockedBy"), kw.get("removeBlockedBy")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
    "background_run":   lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
    "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":              lambda **kw: "Lead does not idle.",
    "claim_task":        lambda **kw: claim_task(kw["task_id"], "lead"),
    "task_bind_worktree": lambda **kw: TASKS.bind_worktree(kw["task_id"], kw["worktree"], kw.get("owner", "")),
    "worktree_create": lambda **kw: WORKTREES.create(kw["name"], kw.get("task_id"), kw.get("base_ref", "HEAD")),
    "worktree_list": lambda **kw: WORKTREES.list_all(),
    "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
    "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
    "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
    "worktree_remove": lambda **kw: WORKTREES.remove(kw["name"], kw.get("force", False), kw.get("complete_task", False)),
    "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
}

# 系统提示词
SYSTEM = f"""
You are a coding agent and a team lead at {WORKDIR} on a Windows Operating System. 
Planning: 
          Use the todo tool to plan multi-step tasks (mark as in_progress when starting, completed when done). 
          Important: Don't use todo tool easily. Todo is only used for complex tasks that may have five or more steps.
          Use background_run for long-running commands.
          Use task + worktree tools for multi-task work.
          For parallel or risky changes: create tasks, allocate worktree lanes, run commands in those lanes, then choose keep/remove for closeout.
          Use worktree_events when you need lifecycle visibility.
Knowledge: Use load_skill to access specialized knowledge for unfamiliar topics. Available skills: {SKILL_LOADER.get_descriptions()}.
Multi-Agent:
          Spawn teammates and communicate via inboxes. 
          Manage teammates with shutdown and plan approval protocols. 
          Teammates are autonomous -- they find work themselves.
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
                print(f"\n> {block.name}:")
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

    # TODO:<<<< worktree worktree ----
    print(f"Repo root for s12: {REPO_ROOT}")
    if not WORKTREES.git_available:
        print("Note: Not in a git repo. worktree_* tools will return errors.")
    # TODO:---- worktree worktree >>>>

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
            # 检查用户输入的指令是否为 "/tasks" (这是一个查看任务看板的特殊指令，不需要发给 LLM 处理)
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                # 根据任务状态获取对应的视觉标记 ([ ] [>] [x])
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                # 如果任务有归属者，则将其格式化为 " @名字" (例如 @alice)，否则为空字符串
                owner = f" @{t['owner']}" if t.get("owner") else ""
                # 在控制台打印出格式化后的任务信息
                # 示例输出: [ ] #5: Fix bug @alice
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
                continue
        # TODO:---- agent_teams:Command-line Debugger >>>>

        history.append({"role": "user","content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list): # 如果是结构化 block(列表)
            for block in response_content:
                if hasattr(block, "text"): # 有 text 属性就打印
                    print()
                    print("----------------------this is the response----------------------")
                    print(block.text)
        print()
"""
产生好奇，s_full肯定集成了TODO和agent_teams，s_full怎么样？
试了试发现没问题。
那么s_full如何处理的？
等看完s10再看s_full。因为s_full把s10和s9混合了。
其实基模的内容也很影响任务成功率。
因为基模好，可能LLM自己就知道todo的固定化顺序可能会污染agent_team的通信。而且在做那些提示词测试的时候，可能成功率也会更高。
可能 share-ai lab 的基模用的比较好。
"""