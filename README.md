# Neutoken BM Platform (接口模型评测控制台)

这是一个通用且功能完善的大语言模型（LLM）基准评测控制台与可视化看板系统。项目采用现代化的前端 UI 交互设计，配合 Flask 后端管理评测任务、并发控制、结果展示以及实时日志输出。

---

## 🌟 核心特性

1. **多维度指标可视化 (Chart.js)**：
   - **精度评测验收得分 (Accuracy Score)**：直观展示各个精度基准测试（如 AIME2025、HLE、SWE-Bench 等）的实际得分与官方基准对照线。
   - **首包时延与上下文长度关系 (TTFT vs Token Size)**：精细展示在不同上下文长度区间内，**TTFT P50** 与 **TTFT P90** 延迟时间变化曲线。

2. **现代化交互与精美设计 (NeuToken Zinc Theme)**：
   - **多主题适配**：完美支持亮色（Light）、暗色（Dark）及跟随系统（System）三大主题，包含顺滑的过渡动画与完美的 UI 覆写。
   - **悬停式主题切换**：支持鼠标悬停下拉展开菜单，包含系统偏好检测状态，并配备防漂移悬停桥接结构。
   - **防抖动卡片布局**：锁定测试描述行高与卡片最小高度，保障测试任务运行时日志更新页面绝对静止无抖动。

3. **安全与控制机制**：
   - **安全访问控制**：支持访问密码保护，通过全屏磨砂玻璃遮罩强制校验，后端在 `before_request` 级别拦截所有 API 并校验请求头与数据流参数。
   - **API 密钥后端脱敏**：前端与网络传输中只暴露脱敏后的密钥（如 `sk-rJQ...wBDW`），明文密钥始终保存在服务器后端。
   - **自定义 API 端点**：允许在配置中在线填写并测试 `api_base` 与 `api_key` 的连通性，自动执行 `/models` / `/chat/completions` 双重连通性测试。
   - **实时日志流**：支持 SSE 协议将评测 stdout 日志实时推送到右侧滑动抽屉终端中。

---

## 📦 依赖要求

- **操作系统**：Linux / macOS / Windows
- **运行环境**：Python 3.8+ (推荐使用 Python 3.12+)
- **核心库依赖**：
  - `Flask` (Web 框架)
  - `Flask-CORS` (解决跨域开发问题)
  - `urllib3` (用于后台进行 API 联通性测试)

---

## 🚀 部署指南

### 1. 复制/拉取项目代码
在服务器终端中，切换至您想要存放项目的目录下，然后将代码克隆或放置在本地：
```bash
git clone https://github.com/Bowen1314/neutoken-bm-platform.git
cd neutoken-bm-platform
```

### 2. 安装 Python 依赖
建议使用虚拟环境（virtualenv）或直接使用系统 Python 安装依赖：
```bash
pip3 install -r requirements.txt
```

### 3. 配置系统属性
在启动之前，您需要创建或修改项目根目录下的 `config.json` 配置文件。

**`config.json` 模板如下：**
```json
{
  "access_password": "您的安全访问密码（若留空则不启用密码锁）",
  "model": {
    "name": "qwen3.7-max",
    "provider": "Neutoken BM Platform",
    "api_base": "https://neutoken.net/v1",
    "api_key": "您的明文接口秘钥",
    "api_key_env": "KIMI_API_KEY",
    "precision": "待确认",
    "max_context": 131072,
    "max_output": 8192,
    "thinking": {
      "type": "disabled"
    }
  },
  "benchmarks": {
     ... (基准评测官方分配置)
  }
}
```

### 4. 启动 Web 服务
我们提供了一个包含环境准备、端口绑定及日志重定向的启动脚本 `start.sh`。

#### A. 在前台直接运行测试（用于调试）：
```bash
chmod +x start.sh
./start.sh
```
服务默认会在端口 **`5001`** 启动。您可以在浏览器中打开 `http://<服务器IP>:5001` 进行访问。

#### B. 在后台持久化运行（推荐）：
为了防止关闭 SSH 终端导致网页服务挂掉，建议使用 `tmux` 在后台持久化运行服务：

1. **新建一个名为 `kimi_benchmark` 的后台会话**：
   ```bash
   tmux new -s kimi_benchmark
   ```
2. **在会话中执行启动脚本**：
   ```bash
   ./start.sh > webui.log 2>&1
   ```
3. **离开（Detach）后台会话**：
   同时按下键盘上的 `Ctrl + B`，然后松开，再按一次 `D` 键即可退出会话。此时服务已安全在后台运行。
4. **以后若需重新进入控制台查看服务状态**：
   ```bash
   tmux attach -t kimi_benchmark
   ```

---

## 🔒 安全须知

1. **防泄漏机制**：
   - 项目自带的 `.gitignore` 已经将 `config.json` 文件、备份文件 `*.bak*`、`*.old` 以及所有产生评测报告和日志的任务输出目录（`results/`、`outputs/`、`logs/`）进行了完全屏蔽。
   - **绝对不要**使用 `git add -f` 强制提交这些目录，以防历史日志中转储的原始 API Key 泄露到公共代码托管平台。

2. **配置修改**：
   - 修改 Base URL 或 API Key 时，后台会对您提交的接口凭证发起连接探测，探测通过方可成功绑定。如果密码开启，保存这些操作也必须通过 `X-Access-Password` 的校验。
