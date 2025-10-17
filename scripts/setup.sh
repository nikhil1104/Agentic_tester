#!/bin/bash
# AI QA Agent - Complete Setup Script
# Supports: macOS, Linux (Ubuntu/Debian)

set -e

echo "=========================================="
echo "AI QA Agent - Setup Script"
echo "=========================================="

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=Mac;;
    *)          PLATFORM="UNKNOWN:${OS}"
esac

echo "Detected platform: $PLATFORM"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ required. Current: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
if [ "$PLATFORM" = "Mac" ] || [ "$PLATFORM" = "Linux" ]; then
    source .venv/bin/activate
else
    echo "Please manually activate: .venv\\Scripts\\activate"
    exit 1
fi

echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Python dependencies installed"

# Install Playwright browsers
echo ""
echo "Installing Playwright browsers..."
playwright install chromium firefox webkit
playwright install-deps

echo "✅ Playwright browsers installed"

# Setup directories
echo ""
echo "Creating directory structure..."
mkdir -p data/vector_store
mkdir -p data/learning_memory
mkdir -p data/visual_baselines
mkdir -p data/visual_diffs
mkdir -p data/scraped_docs
mkdir -p reports
mkdir -p auth
mkdir -p generated_frameworks
mkdir -p logs

echo "✅ Directories created"

# Setup environment file
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cat > .env << 'EOF'
# AI QA Agent Configuration

# OpenAI API
OPENAI_API_KEY=your-openai-api-key-here

# Worker Configuration
WORKER_API_KEY=supersecret123
WORKER_LOG_LEVEL=INFO
WORKER_DEFAULT_TIMEOUT=1800
RUNNER_REPORTS_DIR=./reports
WORKER_ENABLE_METRICS=true
WORKER_METRICS_PORT=9108

# Optional Integrations
PERCY_TOKEN=
SERPER_API_KEY=

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
EOF
    echo "✅ .env file created (please update with your API keys)"
else
    echo "ℹ️  .env file already exists"
fi

# Install Node.js dependencies (if package.json exists)
if [ -f package.json ]; then
    echo ""
    echo "Installing Node.js dependencies..."
    npm install
    echo "✅ Node.js dependencies installed"
fi

# Initialize RAG engine
echo ""
echo "Initializing RAG engine..."
python3 << 'PYTHON'
from modules.rag_engine import RAGEngine
try:
    rag = RAGEngine()
    print("✅ RAG engine initialized")
except Exception as e:
    print(f"⚠️  RAG initialization failed: {e}")
PYTHON

# Run health check
echo ""
echo "Running health check..."
python3 << 'PYTHON'
import sys
try:
    from modules.spec_orchestrator import SpecOrchestrator
    from modules.learning_memory import LearningMemory
    from modules.rag_engine import RAGEngine
    
    print("✅ All core modules imported successfully")
    sys.exit(0)
except Exception as e:
    print(f"❌ Health check failed: {e}")
    sys.exit(1)
PYTHON

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Update .env with your API keys"
    echo "2. Run: python -m modules.spec_orchestrator --requirement 'Test https://example.com'"
    echo "3. Or start worker: uvicorn worker.app:app --reload"
    echo ""
    echo "For MCP integration (Claude Desktop):"
    echo "  python -m modules.mcp_server"
    echo ""
else
    echo ""
    echo "❌ Setup completed with warnings"
    echo "Please check error messages above"
fi
