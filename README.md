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

## 🎯 精度验收测试 (Accuracy Benchmarks) 额外依赖安装指南

平台中的精度验收测试套件包含 9 个主流基准测试。若要运行这些精度评测，除了安装核心库外，还需要根据具体测试项安装以下依赖并配置运行环境：

### 1. SWE-bench 系列测试 (代码修复评测)
*   **虚拟化环境 (Docker)**：必须已安装并运行 Docker 守护进程（Daemon）。底层评测引擎（基于 `evalscope` / `SWE-bench` Harness）会自动调用 Docker API 创建沙箱容器，拉取测试环境镜像并在容器内对目标软件仓库应用补丁并运行集成测试。
    *   **Docker 权限配置**：确保运行测试的非 root 用户已被加入 `docker` 用户组，无需 `sudo` 即可运行 `docker` 命令：
        ```bash
        sudo usermod -aG docker $USER
        # 重启终端或注销重新登录以生效
        ```
*   **Python 依赖库**：
    ```bash
    pip install docker>=7.0.0
    pip install swebench==4.1.0
    ```
    *注意：EvalScope 精度评测中需要精确绑定 `swebench==4.1.0` 版本，请勿安装其他版本。*
*   **网络连通性与镜像拉取**：评测过程中会自动从 Docker Hub 拉取测试环境镜像（如 `swebench/eval-verified` 等）。若服务器在国内或处于无公网环境：
    1. 需配置 Docker 国内加速镜像源或使用代理。
    2. 或预先在有网环境拉取所需的基础镜像，通过 `docker save` 和 `docker load` 导入到测试机中。

### 2. TAU-bench 系列测试 (Agent工具调用评测)
*   **Python 依赖库**：需要安装 Sierra Research 官方的 `tau-bench` 库。您可以通过 Git 链接直接安装：
    ```bash
    pip install git+https://github.com/sierra-research/tau-bench.git
    ```
    或者克隆到本地后进行可编辑模式安装：
    ```bash
    git clone https://github.com/sierra-research/tau-bench.git
    cd tau-bench
    pip install -e .
    ```
*   **模型功能要求**：TAU-bench 主要测试模型的多轮工具调用（Function Calling）能力，需确保在网页端或配置文件中配置的被测模型端点支持并开启了工具调用功能，且配置了有效的 `api_key`。

### 3. HLE (Humanity's Last Exam) 全学科多模态评测
*   **HLE 评测裁判设置**：HLE 基准使用模型本身（或配置中指定的评测模型）作为裁判（Judge Model）进行打分。因此，需确保 API 账户有足够的额度与并发能力，且当前测试模型能够正常响应打分请求。
*   **网络要求**：评测过程中依赖 EvalScope 自动从 ModelScope (魔搭社区) 下载 HLE 评测数据集。若连接魔搭社区较慢，可设置本地缓存目录：
    ```bash
    export MODELSCOPE_CACHE="/path/to/cache"
    ```

### 4. 其他精度基准 (AIME2025/2026, GPQA Diamond, LongBench V2)
*   **依赖安装**：仅需安装基础框架中的 `evalscope`。
*   **数据集来源**：
    *   **GPQA Diamond**：自动从魔搭社区加载 `AI-ModelScope/gpqa_diamond` 仓库。
    *   **AIME2025/AIME2026**：通过 `evalscope` 内置的数据集加载器进行本地或云端加载。
    *   **LongBench V2**：长上下文理解评测，使用内置适配器拉取。
*   请确保服务器能够正常连接网络以进行首次评测数据集的自动下载（国内服务器魔搭社区下载免翻墙，速度较快）。

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
在启动之前，您需要将根目录下的模板配置文件复制为 `config.json`，然后再进行修改：
```bash
cp config.example.json config.json
```
然后编辑 `config.json` 写入您的具体凭证。

**配置文件结构示范如下：**
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

### 4. 配置与修改 API 凭证 (API Base & API Key)

本系统支持**两种方式**来配置和更新您的 API 接口凭证（Base URL 和 API Key）：

#### 方式 A：通过网页端可视化配置（推荐，简单直观）
1. 打开浏览器访问平台控制台，在左侧导航栏的 **“系统配置”** 下点击 **“API 状态信息”**。
2. 在页面右下侧的“修改接口设置”表单中，输入您的 **API 接口端点 (Base URL)** 和 **API 密钥 (API Key)**。
3. 点击 **“保存配置”**。
   - 系统后台会使用您填写的信息，自动发起安全联通性测试（优先调用 `/models` 列出模型，不支持时降级至调用 `/chat/completions` 发送 1 个 token 的测试请求）。
   - **联通性验证成功**：配置将自动写入到宿主机的 `config.json` 文件中，并即时更新系统内存配置生效。
   - **联通性验证失败**：系统将抛出 `绑定失败: [连接报错详情]` 错误提示并拒绝修改，避免配置错误导致后续评测断连。

#### 方式 B：修改本地 `config.json` 配置文件
1. 使用文本编辑器直接修改项目根目录下的 `config.json` 文件。
2. 更改 `model` 对象下的 `"api_base"` 和 `"api_key"` 的属性值。
3. 修改后，必须**重启后端 Flask 服务**才能载入并生效。

### 5. 安全访问密码配置与重置

为了防止未授权的人员访问评测系统或控制评测任务，您可以在系统根目录下的 `config.json` 中配置安全访问密码：

#### A. 启用与修改访问密码
1. 编辑项目根目录下的 `config.json`，在最外层（与 `model` 同级）添加或修改 `"access_password"` 字段：
   ```json
   {
     "access_password": "您的强访问密码（例如: CGxQa0HC*9gdtq#N9a）",
     "model": { ... }
   }
   ```
2. 保存配置文件，并**重启后端 Flask 服务**以重载配置生效。
3. 服务重启后，任何用户访问控制台页面都必须输入对应的密码以通过前端磨砂玻璃登录弹窗。所有 API 接口（包括日志流）在后端都会拦截验证，没有密码的一律返回 `401 Unauthorized` 报错，确保安全。
4. 若需要关闭访问密码，只需将 `"access_password"` 的值设为空字符串 `""` 并重启服务即可。

#### B. 忘记密码时如何查看或重置
如果您在使用过程中忘记了密码，可以直接登录服务器查看或重置：
1. **查看明文密码**：在服务器终端切换到项目目录，运行以下命令过滤显示当前密码：
   ```bash
   cat config.json | grep access_password
   ```
2. **重置密码**：使用 `nano config.json` 或 `vi config.json` 直接编辑该文件并更新 `"access_password"` 后面的数值，随后重启 Flask 网页服务即可生效。

### 6. 启动 Web 服务
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
