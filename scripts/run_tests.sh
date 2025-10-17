#!/bin/bash
# Run all test suites

set -e

echo "=========================================="
echo "AI QA Agent - Test Suite"
echo "=========================================="

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Run unit tests
echo ""
echo "Running unit tests..."
pytest tests/ \
    --cov=modules \
    --cov=worker \
    --cov-report=html \
    --cov-report=term \
    --junitxml=junit/test-results.xml \
    -v

# Run integration tests
echo ""
echo "Running integration tests..."
pytest tests/integration/ \
    --timeout=600 \
    -v

# Generate coverage report
echo ""
echo "Generating coverage report..."
coverage html
echo "✅ Coverage report generated: htmlcov/index.html"

echo ""
echo "=========================================="
echo "✅ All tests passed!"
echo "=========================================="
