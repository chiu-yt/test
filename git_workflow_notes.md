# Git 学习笔记（适合当前环境）

> 适用环境：GitHub **HTTPS 不通**、**SSH 可通**；本地电脑负责开发和使用 Codex，服务器负责拉代码和运行程序。

---

## 你的环境结论

- GitHub **HTTPS 不通**
- GitHub **SSH 可通**
- 远程仓库统一使用 **SSH 地址**：

```bash
git@github.com:chiu-yt/test.git
```

不要使用：

```bash
https://github.com/chiu-yt/test.git
```

---

# 1. Git 的核心概念

## 工作区（Working Directory）
你当前正在编辑的项目文件夹。

## 暂存区（Staging Area）
你准备放进下一次提交的改动集合。

## 本地仓库（Local Repository）
你电脑或服务器上的 Git 历史记录。

## 远程仓库（Remote Repository）
GitHub 上的仓库。

## 提交（Commit）
一次版本快照。

## 分支（Branch）
一条开发线。

## origin
默认的远程仓库名，通常就是 GitHub 上那个仓库。

---

# 2. 一次性初始化流程

这些通常只做一次。

## 2.1 配置 Git 用户名和邮箱

```bash
git config --global user.name "chiu-yt"
git config --global user.email "15255742990@163.com"
```

### 命令作用

- `git config`：设置 Git 配置
- `--global`：对当前用户全局生效
- `user.name`：提交记录中显示的名字
- `user.email`：提交记录中显示的邮箱

查看配置：

```bash
git config --global --list
```

---

## 2.2 验证 SSH 是否可用

```bash
ssh -T git@github.com
```

### 命令作用

- `ssh`：通过 SSH 连接远程主机
- `-T`：不打开远程 shell，只做认证测试
- `git@github.com`：GitHub 的 Git SSH 入口

成功时通常会看到：

```text
Hi chiu-yt! You've successfully authenticated, but GitHub does not provide shell access.
```

这说明：
- SSH 已通
- 公钥认证成功
- 可以使用 GitHub SSH 仓库地址

---

# 3. 第一次克隆仓库

## 3.1 在本地电脑克隆

```bash
git clone git@github.com:chiu-yt/test.git
cd test
```

## 3.2 在服务器克隆

```bash
git clone git@github.com:chiu-yt/test.git
cd test
```

### 命令作用

- `git clone`：把远程仓库完整复制到本地
- `cd test`：进入项目目录

---

# 4. 最常用工作流：本地改代码 → push → 服务器 pull → 运行

这是你以后最常用的流程。

## 4.1 本地只修改一个文件

```bash
cd ~/test
git status
git diff path/to/file
git add path/to/file
git commit -m "修改某个文件"
git push origin main
```

### 每条命令的作用

#### `cd ~/test`
进入项目目录。

#### `git status`
查看当前仓库状态，包括：
- 哪些文件修改了
- 哪些文件未跟踪
- 哪些文件已暂存
- 当前所在分支

#### `git diff path/to/file`
查看某个文件的具体改动。

#### `git add path/to/file`
只把这个文件加入暂存区，表示这次提交只准备提交这个文件。

#### `git commit -m "修改某个文件"`
创建一次提交。

- `commit`：提交暂存区内容
- `-m`：直接写提交说明（message）

#### `git push origin main`
把本地 `main` 分支推送到远程仓库 `origin`。

- `push`：上传提交到远程
- `origin`：远程仓库名
- `main`：分支名

---

## 4.2 本地修改多个文件

```bash
cd ~/test
git status
git diff
git add .
git commit -m "实现某个功能"
git push origin main
```

### 每条命令的作用

#### `git diff`
查看所有未暂存改动。

#### `git add .`
把当前目录及子目录里所有改动加入暂存区。

> 注意：如果你只想提交一个文件，不要用 `git add .`。

---

## 4.3 服务器更新代码

```bash
cd ~/test
git pull --ff-only
```

### 命令作用

#### `git pull --ff-only`
从远程拉取最新代码并更新当前分支。

它本质上相当于：

```bash
git fetch
git merge
```

但加上 `--ff-only` 后，只允许“快进式更新”：
- 如果服务器本地没有额外提交，就直接更新
- 如果会产生复杂合并，则直接报错
- 适合服务器这种“只运行、不开发”的机器

---

## 4.4 服务器运行程序

# 5. 空仓库第一次提交

适用于：GitHub 上刚创建了一个空仓库。

```bash
mkdir test
cd test
git init
git config user.name "chiu-yt"
git config user.email "15255742990@163.com"
echo hello > hello.txt
git add hello.txt
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:chiu-yt/test.git
git push -u origin main
```

### 每条命令的作用

#### `mkdir test`
创建项目目录。

#### `git init`
把当前目录初始化成一个 Git 仓库。

#### `echo hello > hello.txt`
创建测试文件。

#### `git branch -M main`
把当前分支强制重命名为 `main`。

#### `git remote add origin git@github.com:chiu-yt/test.git`
为当前仓库添加远程仓库 `origin`。

#### `git push -u origin main`
把本地 `main` 推到远程，并建立跟踪关系。

- `-u`：设置 upstream，之后可以直接 `git push` / `git pull`

---

# 6. 每天开工前的标准流程

```bash
cd ~/test
git status
git pull --ff-only
git log --oneline -5
```

### 命令作用

#### `git log --oneline -5`
查看最近 5 次提交。

- `log`：查看提交历史
- `--oneline`：每条提交一行
- `-5`：只看最近 5 条

---

# 7. 分支开发流程（推荐）

适用于：你不想直接在 `main` 上开发，而是先在功能分支上改。

## 7.1 创建并切换到新分支

```bash
cd ~/test
git pull --ff-only
git checkout -b feature-1
```

### 命令作用

- `git checkout -b feature-1`：创建并切换到新分支 `feature-1`

---

## 7.2 在新分支开发并提交

```bash
git status
git add .
git commit -m "完成某个功能"
git push -u origin feature-1
```

### 命令作用

- `git push -u origin feature-1`：把功能分支推到远程并建立跟踪关系

---

## 7.3 合并回 main

```bash
git checkout main
git pull --ff-only
git merge feature-1
git push origin main
```

### 命令作用

- `git checkout main`：切回主分支
- `git merge feature-1`：把功能分支合并到当前分支
- `git push origin main`：把合并后的主分支推到远程

---

## 7.4 删除已完成分支

本地删除：

```bash
git branch -d feature-1
```

远程删除：

```bash
git push origin --delete feature-1
```

### 命令作用

- `git branch -d feature-1`：删除本地分支
- `git push origin --delete feature-1`：删除远程分支

---

# 8. 查看改动和历史

## 8.1 查看当前状态

```bash
git status
```

## 8.2 查看未暂存改动

```bash
git diff
```

## 8.3 查看某个文件的改动

```bash
git diff path/to/file
```

## 8.4 查看已暂存但还没提交的改动

```bash
git diff --cached
```

## 8.5 查看提交历史

```bash
git log
git log --oneline
git log --oneline -10
```

## 8.6 查看某个文件的历史

```bash
git log -- path/to/file
```

## 8.7 查看某次提交改了什么

```bash
git show 提交ID
```

例如：

```bash
git show a327410
```

---

# 9. 只提交部分改动

适用于：你改了很多文件，但这次只想提交其中一部分。

```bash
git status
git add file1
git add dir/file2
git commit -m "只提交部分改动"
git push origin main
```

### 核心思想

只有 `git add` 进去的文件，才会进入这次提交。

---

# 10. 放弃改动

> 注意：这部分命令可能导致代码丢失，操作前一定要确认。

## 10.1 放弃某个文件的未提交改动

```bash
git restore path/to/file
```

## 10.2 放弃所有未暂存改动

```bash
git restore .
```

## 10.3 把已暂存文件移出暂存区

```bash
git restore --staged path/to/file
```

## 10.4 丢弃所有未提交改动（危险）

```bash
git reset --hard
```

### 命令作用

- `git restore`：恢复文件内容
- `--staged`：只取消暂存，不删除文件改动
- `git reset --hard`：回到最近一次提交状态，危险命令

---

# 11. 临时保存改动（stash）

适用于：代码改到一半，不想提交，但又要先切分支或先拉代码。

## 11.1 临时保存当前改动

```bash
git stash
```

## 11.2 查看 stash 列表

```bash
git stash list
```

## 11.3 恢复最近一次 stash

```bash
git stash pop
```

## 11.4 恢复但不删除 stash

```bash
git stash apply
```

### 命令作用

- `git stash`：把当前未提交改动临时收起来
- `git stash pop`：恢复并删除最近一次 stash
- `git stash apply`：恢复但保留 stash 记录

---

# 12. 服务器上不小心改了代码怎么办

## 12.1 先查看状态

```bash
git status
```

## 12.2 如果有本地改动，先收起来，再拉远程更新

```bash
git stash
git pull --ff-only
git stash pop
```

### 注意

如果你改动的内容和远程更新冲突，`git stash pop` 时可能需要手动解决冲突。

---

# 13. 解决冲突

## 13.1 查看冲突文件

```bash
git status
```

## 13.2 手动编辑文件

冲突文件中通常会出现：

```text
<<<<<<< HEAD
你的内容
=======
远程内容
>>>>>>> 分支名
```

你需要：
- 手工保留正确内容
- 删除这些冲突标记

## 13.3 解决完后重新提交

```bash
git add 冲突文件
git commit -m "resolve conflict"
```

---

# 14. 回滚流程

## 14.1 回滚最近一次提交，但保留改动在暂存区

```bash
git reset --soft HEAD~1
```

## 14.2 回滚最近一次提交，取消暂存，但保留改动在工作区

```bash
git reset --mixed HEAD~1
```

## 14.3 回滚最近一次提交，连改动一起丢掉（危险）

```bash
git reset --hard HEAD~1
```

## 14.4 已经 push 到远程，安全撤销某次提交

```bash
git revert 提交ID
```

例如：

```bash
git revert a327410
```

### 命令作用

- `reset`：移动当前分支指针
- `--soft`：保留暂存区和工作区
- `--mixed`：保留工作区，取消暂存
- `--hard`：全部丢弃
- `revert`：新增一个反向提交来撤销指定提交

> 已经 push 的提交，优先用 `git revert`，不要轻易用 `git reset --hard` 后强推。

---

# 15. 远程仓库相关操作

## 15.1 查看远程仓库

```bash
git remote -v
```

## 15.2 新增远程仓库

```bash
git remote add origin git@github.com:chiu-yt/test.git
```

## 15.3 修改远程仓库地址

```bash
git remote set-url origin git@github.com:chiu-yt/test.git
```

## 15.4 删除远程仓库

```bash
git remote remove origin
```

### 命令作用

- `remote -v`：查看远程仓库详细地址
- `remote add`：添加新远程
- `remote set-url`：修改远程地址
- `remote remove`：删除远程

---

# 16. 先拉远程但暂不合并

适用于：你想先看看远程更新了什么。

```bash
git fetch
git log --oneline HEAD..origin/main
```

### 命令作用

- `git fetch`：只下载远程更新，不合并
- `git log --oneline HEAD..origin/main`：查看远程分支比当前多了哪些提交

---

# 17. 分支切换相关操作

## 17.1 查看本地分支

```bash
git branch
```

## 17.2 查看本地和远程分支

```bash
git branch -a
```

## 17.3 切换分支

```bash
git checkout main
```

或者：

```bash
git switch main
```

## 17.4 创建并切换到新分支

```bash
git switch -c feature-1
```

等价于：

```bash
git checkout -b feature-1
```

---

# 18. 标签 tag 流程

适用于：给某个版本打标记，比如 `v1.0.0`。

## 18.1 创建标签

```bash
git tag v1.0.0
```

## 18.2 查看标签

```bash
git tag
```

## 18.3 推送某个标签

```bash
git push origin v1.0.0
```

## 18.4 推送所有标签

```bash
git push origin --tags
```

---

# 19. 最适合你当前环境的推荐流程

## 本地开发机
只负责：
- 用 Codex 改代码
- 本地测试
- 提交
- 推送

推荐命令：

```bash
cd ~/test
git pull --ff-only
git add path/to/file
git commit -m "描述修改"
git push origin main
```

## 服务器
只负责：
- 拉最新代码
- 运行程序
- 不直接改业务代码

推荐命令：

```bash
cd ~/test
git pull --ff-only
python app.py
```

根据你的项目类型把 `python app.py` 换成实际启动命令。

---

# 20. 最常见报错及含义

## `Permission denied (publickey)`
含义：
- SSH 网络已通
- 但 GitHub 不认可你的公钥

处理：
- 检查 SSH key 是否已加到 GitHub
- 检查是否已 `ssh-add` 到 agent

## `src refspec main does not match any`
含义：
- 本地还没有任何提交
- 或当前分支不是 `main`

处理：
- 先完成一次 `commit`
- 检查 `git branch`

## `Repository not found`
含义：
- 仓库地址错了
- 或没有访问权限

处理：
- 检查 `git remote -v`
- 检查仓库是否存在

## `Your local changes would be overwritten by merge`
含义：
- 本地有未提交改动
- 不能直接 `pull`

处理：

```bash
git stash
git pull --ff-only
git stash pop
```

---

# 21. 最值得背下来的命令

查看状态：

```bash
git status
```

查看改动：

```bash
git diff
git diff path/to/file
```

提交改动：

```bash
git add path/to/file
git commit -m "message"
```

推送：

```bash
git push origin main
```

拉取：

```bash
git pull --ff-only
```

看历史：

```bash
git log --oneline -5
```

临时保存：

```bash
git stash
git stash pop
```

放弃改动：

```bash
git restore path/to/file
git reset --hard
```

---

# 22. 最终一句话总结

你的最小闭环就是：

**本地用 Codex 改代码 → 本地 commit → 本地 push → 服务器 pull → 服务器运行**

最小命令版：

## 本地

```bash
cd ~/test
git add path/to/file
git commit -m "修改说明"
git push origin main
```

## 服务器

```bash
cd ~/test
git pull --ff-only
python app.py
```

把 `python app.py` 替换成你的实际启动命令即可。

---

# 23. 针对当前 OpenPCDet 项目的上传规则

你的当前使用方式是：

- 本地电脑：改代码、看 diff、commit、push
- 远程服务器：pull 最新代码、训练、测试、保存输出

因此远程仓库里应该只保存“代码和配置”，不保存“服务器运行后生成的内容”。

## 23.1 应该上传的内容

这些内容通常应该提交到 Git：

- `pcdet/`：核心源码
- `tools/`：训练、测试、TTA、demo 脚本
- `tools/cfgs/`：实验配置文件 `*.yaml`
- `setup.py`
- `requirements.txt`
- 你自己新增的辅助脚本，例如数据划分脚本、分析脚本
- `README.md`、实验记录、工作流说明等文档

一句话理解：

**凡是服务器拉下来后，为了正确运行而必须存在的代码、配置、脚本，都应该上传。**

## 23.2 不应该上传的内容

这些内容通常不要提交到 Git：

- `data/`：数据集、标注、infos
- `output/`：训练输出目录
- `results/`：评测结果目录
- `tensorboard/`：可视化日志目录
- `wandb/`：实验跟踪目录
- `*.pth`、`*.ckpt`：模型权重和 checkpoint
- `build/`、`dist/`、`*.egg-info/`：构建产物
- `*.so`：本机或服务器编译出来的扩展文件
- `__pycache__/`、`*.pyc`：Python 缓存
- `.ipynb_checkpoints/`、`.DS_Store`：编辑器和系统垃圾文件
- `venv/`、`.venv/`、`env/`：虚拟环境目录

一句话理解：

**凡是服务器训练、测试、编译后自动生成的内容，默认都不要上传。**

## 23.3 当前项目建议的 `.gitignore` 思路

当前项目建议忽略这些路径或类型：

```gitignore
__pycache__/
*.pyc
*.pyo
build/
dist/
*.egg-info/
*.egg
*.so
.ipynb_checkpoints/
.DS_Store
venv/
.venv/
env/
data/
**/data/
output/
**/output/
results/
**/results/
tensorboard/
**/tensorboard/
wandb/
**/wandb/
*.log
*.pth
*.ckpt
*.zip
*.tar
*.tar.gz
```

## 23.4 每次提交前的最小检查

在本地提交前，建议固定做这几步：

```bash
git status
git diff
git add path/to/file
git commit -m "描述这次修改"
git push origin main
```

如果你看到下面这些路径出现在 `git status` 里，就先不要提交，先检查 `.gitignore`：

- `data/...`
- `output/...`
- `results/...`
- `tensorboard/...`
- `wandb/...`
- `build/...`
- `*.pth`
- `*.so`

## 23.5 服务器的推荐用法

服务器最好只做两件事：

- `git pull --ff-only`
- 运行训练或测试命令

尽量不要在服务器直接修改业务代码，这样可以避免：

- pull 时冲突
- 本地和服务器代码不一致
- 不小心把训练产物也提交进 Git

适合你的最小闭环仍然是：

**本地改代码 -> 本地 commit -> 本地 push -> 服务器 pull -> 服务器训练/测试**
