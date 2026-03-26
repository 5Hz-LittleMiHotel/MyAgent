**总结**

SKILL.md首先扮演了提示词的角色，给EcoA一份智能体构建指南，教它怎么极简、高效地搭建自己想要的子智能体，并且使用reference告诉EcoA有哪些资源可以用。当需要时，EcoA会通过SKILL.md读取三个案例（包括Agent 长什么样，Agent 能定义哪些工具，Agent 如何分工），并通过init_agent.py去制造属于它的子智能体团队，来完成任务。


**SKILL.md 的两层作用**

第一层是**提示词**：正文部分（核心哲学、三要素、设计思维、反模式等）直接塑造 EcoA 的思维方式——告诉它怎么想、怎么取舍、信任模型而不是过度工程化。

第二层是**资源索引**：Resources 部分告诉 EcoA "当你需要动手时，去哪里找参考"。

**四个资源文件的分工**

```
agent-philosophy.md   → 为什么这样做（世界观）
minimal-agent.py      → 骨架长什么样（起点）
tool-templates.py     → 工具怎么定义（扩展点）
subagent-pattern.py   → 子Agent怎么协作（规模化）
init_agent.py         → 以上一切怎么落地到磁盘（执行）
```

**有一点值得补充**

SKILL.md 里有一句话其实是整个设计的灵魂：

> Most agents never need to go beyond Level 2.

这不只是一个观察，它也是给 EcoA 的**行为约束**——不要过度创建子 Agent，不要把简单问题复杂化。所以这个 SKILL 既授予了 EcoA 构建子 Agent 团队的能力，同时也在提示词层面告诉它**克制地使用**这个能力。能力和约束被封装在同一个文件里，这个设计本身就很elegant。