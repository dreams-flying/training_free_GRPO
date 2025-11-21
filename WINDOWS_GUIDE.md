# Windows 部署指南 / Windows Deployment Guide

本指南专门针对在 Windows 平台上部署和运行 Training-Free GRPO 的用户。

This guide is specifically for users deploying and running Training-Free GRPO on Windows.

---

## 已解决的 Windows 兼容性问题 / Fixed Windows Compatibility Issues

我们已经修复了以下 Windows 平台的常见问题：

✅ **多进程句柄错误** - `OSError: [WinError 6] 句柄无效`
✅ **dotenv 解析错误** - `python-dotenv could not parse statement`
✅ **API 速率限制** - `openai.RateLimitError: local_rate_limited`
✅ **resource 模块不可用** - Windows 上自动跳过内存限制功能

---

## 快速开始 / Quick Start

### 1. 拉取最新代码 / Pull Latest Code

```bash
git pull origin claude/fix-mult-import-error-01VbU29ZSU5b8cV8Q4oquYEj
```

### 2. 创建环境变量文件 / Create .env File

```bash
# 复制模板文件
copy .env.example .env

# 使用记事本编辑（确保文件编码为 UTF-8）
notepad .env
```

**重要：.env 文件格式要求**
- 使用 `KEY=VALUE` 格式，**不要加引号**
- 等号两边**不要有空格**
- 每行一个配置项
- 以 `#` 开头的是注释

示例：
```ini
# 正确格式 ✅
UTU_LLM_API_KEY=sk-your-api-key-here
UTU_LLM_MODEL=deepseek-chat

# 错误格式 ❌
UTU_LLM_API_KEY="sk-your-api-key-here"  # 不要加引号
UTU_LLM_API_KEY = sk-your-api-key-here  # 等号两边不要空格
```

### 3. 运行训练（保守配置）/ Run Training (Conservative Settings)

```bash
# Math domain - 降低并发数避免速率限制
python training_free_grpo/train.py ^
    --mode agent ^
    --domain math ^
    --experiment_name windows_test ^
    --dataset DAPO-Math-17k ^
    --dataset_truncate 50 ^
    --epochs 2 ^
    --batchsize 10 ^
    --grpo_n 3 ^
    --rollout_concurrency 3 ^
    --rollout_temperature 0.7 ^
    --task_timeout 1800
```

**注意：** Windows 命令行使用 `^` 续行符，不是 `\`

### 4. 运行评估 / Run Evaluation

```bash
python training_free_grpo/main.py ^
    --mode agent ^
    --domain math ^
    --experiment_name windows_eval ^
    --dataset AIME24 ^
    --rollout_concurrency 5 ^
    --pass_k 5
```

---

## Windows 特定注意事项 / Windows-Specific Notes

### 1. 多进程支持 / Multiprocessing Support

Windows 使用 `spawn` 模式创建子进程，与 Linux 的 `fork` 不同。已在以下文件中添加兼容性支持：

- `training_free_grpo/main.py`
- `training_free_grpo/train.py`
- `training_free_grpo/web/dataset.py`
- `utu/tools/python_executor_toolkit.py` ⭐ **最关键的修复**

### 2. 内存限制功能 / Memory Limit Feature

`resource` 模块在 Windows 上不可用，因此内存限制功能会自动跳过。这不影响正常使用。

### 3. 路径分隔符 / Path Separators

代码已自动处理路径分隔符，无需手动修改。Python 的 `os.path.join()` 会自动适配 Windows。

### 4. 命令行参数 / Command Line Arguments

Windows PowerShell 和 CMD 的续行符不同：

**CMD (命令提示符):**
```cmd
python script.py ^
    --arg1 value1 ^
    --arg2 value2
```

**PowerShell:**
```powershell
python script.py `
    --arg1 value1 `
    --arg2 value2
```

---

## 常见问题排查 / Troubleshooting

### 问题 1: 仍然出现句柄错误 / Still Getting Handle Error

**症状:**
```
OSError: [WinError 6] 句柄无效
```

**解决方案:**
1. 确保已拉取最新代码
2. 检查是否有其他 Python 进程在运行，关闭它们
3. 重启命令行窗口
4. 如果使用 Anaconda，尝试创建新的虚拟环境

```bash
# 创建新环境
conda create -n grpo python=3.11
conda activate grpo
pip install -r requirements.txt
```

### 问题 2: dotenv 解析错误 / dotenv Parse Error

**症状:**
```
python-dotenv could not parse statement starting at line X
```

**解决方案:**
1. 确保 `.env` 文件存在（从 `.env.example` 复制）
2. 检查文件格式：不要使用引号，不要有多余空格
3. 确保文件编码为 UTF-8（不是 UTF-8 BOM）
4. 删除所有空行

### 问题 3: 速率限制错误 / Rate Limit Error

**症状:**
```
openai.RateLimitError: local_rate_limited
```

**解决方案:**
1. 大幅降低 `--rollout_concurrency` 参数：

| API 等级 | 推荐值 |
|---------|--------|
| 免费版 | 2-3 |
| 基础版 | 5-10 |
| 专业版 | 20-50 |

2. 代码已实现自动重试（指数退避），通常会自动恢复
3. 如果仍然失败，等待几分钟后重试
4. 参考 `RATE_LIMIT_GUIDE.md` 获取详细说明

### 问题 4: 模块导入错误 / Module Import Error

**症状:**
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案:**
```bash
# 使用 uv 同步环境（推荐）
uv sync

# 或使用 pip 安装
pip install -r requirements.txt

# 如果使用 Anaconda
conda install --file requirements.txt
```

### 问题 5: API 密钥未设置 / API Key Not Set

**症状:**
```
ValueError: Environment variable UTU_LLM_API_KEY is not set
```

**解决方案:**
1. 确保 `.env` 文件在项目根目录
2. 检查文件内容，确保包含所有必需的变量：
   - `UTU_LLM_TYPE`
   - `UTU_LLM_MODEL`
   - `UTU_LLM_BASE_URL`
   - `UTU_LLM_API_KEY`
3. 重启 Python 进程/命令行窗口

---

## 性能优化建议 / Performance Tips

### 1. 使用 SSD / Use SSD

将项目放在 SSD 上可以显著提高数据加载速度。

### 2. 调整并发数 / Adjust Concurrency

根据你的 CPU 核心数和 API 限制调整：

```bash
# 查看 CPU 核心数
python -c "import os; print(os.cpu_count())"

# 建议并发数 = min(CPU核心数, API限制)
```

### 3. 使用虚拟环境 / Use Virtual Environment

建议使用 Conda 或 venv 创建独立环境，避免依赖冲突。

---

## Web Domain 特殊说明 / Web Domain Notes

Web domain 需要额外的 API 密钥：

```ini
# .env 文件中添加
SERPER_API_KEY=your-serper-key
JINA_API_KEY=your-jina-key
```

获取方式：
- Serper: https://serper.dev/
- Jina: https://jina.ai/

---

## 获取帮助 / Getting Help

如果遇到其他问题：

1. 查看 `README.md` - 基础使用说明
2. 查看 `RATE_LIMIT_GUIDE.md` - 速率限制详细指南
3. 检查 GitHub Issues - 搜索类似问题
4. 创建新 Issue - 提供详细的错误信息和环境信息

**提供信息时请包含：**
- Windows 版本（Win 10/11）
- Python 版本
- 完整的错误堆栈信息
- 使用的命令和配置

---

## 测试清单 / Testing Checklist

在开始大规模训练前，建议先运行小规模测试：

```bash
# 最小测试 - Math domain
python training_free_grpo/main.py ^
    --mode agent ^
    --domain math ^
    --experiment_name test ^
    --dataset DAPO-Math-17k ^
    --dataset_truncate 5 ^
    --rollout_concurrency 2 ^
    --pass_k 1

# 如果成功，逐步增加规模
```

---

## 版本历史 / Version History

- **v1.3** (2025-01-XX) - 修复 PythonExecutorToolkit 多进程问题
- **v1.2** (2025-01-XX) - 添加速率限制智能处理
- **v1.1** (2025-01-XX) - 修复 .env 解析问题
- **v1.0** (2025-01-XX) - 初始 Windows 兼容性修复

---

## 致谢 / Acknowledgments

感谢所有在 Windows 上测试和报告问题的用户！

Thank you to all users who tested and reported issues on Windows!
