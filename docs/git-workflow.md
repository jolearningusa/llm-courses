# Git 主干开发流程

## 目标流程

本仓库采用 Trunk-Based Development：只有一个长期分支 `main`，它必须始终处于可发布状态。
日常变更使用短生命周期分支，不建立 `develop`、`staging` 或长期 release 分支。

```text
main ──┬────────────── squash merge ──────────────> (删除分支)
       └─ content/agentos-rust ─ commits ─ PR
```

标准流程：

1. 从最新 `origin/main` 创建分支：`course/*`（新课程或课程大改）、`content/*`（幻灯片、实验、
   参考资料）、`fix/*`、`docs/*` 或 `chore/*`。被指定分支名的 agent 会话（如 `claude/*`）使用指定名称。
2. 分支应在一至两个工作日内合并；较大的课程拆成多个 PR（大纲 → 各模块 → 实验）。
3. push 前先 fetch `origin/main` 并 rebase，完成 [AGENTS.md](../AGENTS.md) 中的本地检查。
4. 按 `.github/pull_request_template.md` 填写 PR，目标分支为 `main`。
5. 通过 squash merge 进入 `main`，随后删除分支（远端由 GitHub 自动删除，本地手动删除）。
6. 不允许直接 push、force-push 或在 `main` 上修复。

**一次只做一个 PR**：用户一次给出多个任务时，当前 PR squash merge 进 `main` 之后，
再从最新 `main` 开下一个任务。不要把 PR 叠在另一个未合并的分支上；已合并的分支不再复用。

## GitHub 仓库设置

在 `Settings → General → Pull Requests`：

- 只启用 **Allow squash merging**，关闭 merge commit 和 rebase merging。
- 勾选 **Automatically delete head branches**（合并后自动删除分支）。

在 `Settings → Rules → Rulesets` 为 `main` 创建 ruleset：

- Require a pull request before merging；单人阶段可以不要求 approval，但仍禁止直接 push。
- Block force pushes、Restrict deletions、Require linear history。
- 以后加入 CI（如链接检查）后，再加入 Require status checks。
