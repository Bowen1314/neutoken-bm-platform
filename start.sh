#!/bin/bash
# 接口模型评测系统启动脚本
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════"
echo "  接口模型评测验收系统"
echo "═══════════════════════════════════════════════"

# Check API key
if [ -z "$KIMI_API_KEY" ]; then
    echo "⚠️  KIMI_API_KEY 未设置"
    echo "   请设置: export KIMI_API_KEY=your-key-here"
    echo "   或使用模拟模式继续..."
    echo ""
fi

# Install deps if needed
if ! python3.12 -c "import flask" 2>/dev/null; then
    echo "📦 安装依赖..."
    pip3 install -r requirements.txt --quiet
fi

PORT=${PORT:-5001}
echo "🚀 启动 WebUI: http://localhost:$PORT"
echo ""
echo "  - 并行测试(⚡): 可同时运行多个"
echo "  - 互斥测试(🔒): 同时只能运行一个"
echo "  - 设置 API Key 后可执行真实测试"
echo "  - 未设置 API Key 时使用模拟模式"
echo ""

PORT=$PORT python3.12 webui/app.py
