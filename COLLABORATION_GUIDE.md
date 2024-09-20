# GitHub Flow 协作开发手册

## 1. 项目负责人职责
作为项目负责人，你负责确保整个项目的顺利运行，包括分支管理、代码审查和最终代码合并。以下是你在 GitHub Flow 中的具体职责和操作步骤：

### 1.1 分支使用
- **`main` 分支**：`main` 是唯一稳定的分支，所有开发者的最终代码都会通过 Pull Request 合并到 `main`。任何直接修改 `main` 分支的操作都应该避免。
- **功能分支 (Feature branches)**：每个新增功能或者 bug 修复都必须在单独的分支上进行开发。开发人员将从 `main` 创建新的分支，如 `feature/feature-name` 或 `bugfix/bug-name`。

### 1.2 代码审查 (Code Review)
- **审核 PR**：当开发人员完成功能或 bug 修复时，会发起 Pull Request (PR) 请求将他们的分支合并到 `main`。你的职责是仔细审查 PR，包括：
  1. **代码质量**：检查代码是否清晰、结构合理，并遵守项目的代码风格规范。
  2. **功能测试**：确保新增功能或修复的 bug 在本地或测试环境中能够正常运行。
  3. **讨论和建议**：对于代码中的任何问题或改进点，可以通过评论在 PR 中与开发人员沟通。
- **合并 PR**：如果代码符合项目标准并且测试通过，你可以点击“Merge Pull Request”按钮将其合并到 `main`。建议使用 **Squash and Merge** 合并策略，这样可以保持 `main` 分支历史记录的简洁。
- **发布版本**：每四个月发布一个版本。确保在每个版本发布前，合并所有功能和 bug 修复，并在 GitHub Releases 中更新版本信息和变更日志。

## 2. 开发人员职责
作为项目中的开发人员，你的主要任务是按照项目需求开发新功能或修复现有问题。以下是你的具体职责和操作步骤：

### 2.1 功能开发步骤
当你负责开发一个新功能时，请遵循以下步骤：

1. **从 `main` 创建一个新分支**：
   - 在开始任何开发之前，从 `main` 创建一个新的功能分支，命名为 `feature/feature-name`：
     ```bash
     git checkout main
     git pull origin main
     git checkout -b feature/feature-name
     ```

2. **进行开发并定期提交**：
   - 在新分支上开发功能，尽量保持小而频繁的提交，以便于回溯和审查：
     ```bash
     git add .
     git commit -m "Implement feature: description"
     ```

3. **推送分支到 GitHub**：
   - 将分支推送到远程仓库：
     ```bash
     git push origin feature/feature-name
     ```

4. **发起 Pull Request**：
   - 在 GitHub 上发起一个 Pull Request，选择合并到 `main`，并在 PR 描述中清楚写明所实现的功能和测试情况。

5. **等待代码审查**：
   - 项目负责人和其他团队成员会对你的 PR 进行代码审查，等待他们的反馈。如果有任何建议或问题需要修改，你需要进行相应的修改并再次提交。

6. **完成合并**：
   - 当 PR 审查通过后，项目负责人会合并你的分支。合并后，及时删除本地和远程的功能分支。

### 2.2 Bug 修复步骤
当你负责修复一个 bug 时，请遵循以下步骤：

1. **从 `main` 创建一个 bugfix 分支**：
   - 创建一个新的分支命名为 `bugfix/bug-name`，从 `main` 基础上进行修复：
     ```bash
     git checkout main
     git pull origin main
     git checkout -b bugfix/bug-name
     ```

2. **进行修复并提交**：
   - 修复问题后，定期提交你的改动：
     ```bash
     git add .
     git commit -m "Fix bug: description"
     ```

3. **推送分支到 GitHub**：
   - 将分支推送到远程仓库：
     ```bash
     git push origin bugfix/bug-name
     ```

4. **发起 Pull Request 并等待审查**：
   - 与功能开发相同，发起 PR，描述修复内容并等待项目负责人进行审查。

5. **完成合并**：
   - 审查通过后，PR 会被合并到 `main`，你可以删除本地和远程的 bugfix 分支。

### 2.3 本地分支管理
- 定期同步 `main`：确保在每次开发新功能或修复之前，先同步本地的 `main` 分支：
  ```bash
  git checkout main
  git pull origin main
