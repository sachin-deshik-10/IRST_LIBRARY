#!/bin/bash
# IRST Library - Run Tests Script

set -e

echo "🧪 Running IRST Library Test Suite"
echo "=================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest pytest-cov pytest-xdist
fi

# Run different test categories
echo "📋 Test Categories:"
echo "  1. Unit Tests"
echo "  2. Integration Tests" 
echo "  3. Performance Tests"
echo "  4. All Tests"
echo ""

read -p "Select test category (1-4): " choice

case $choice in
    1)
        echo "🔬 Running Unit Tests..."
        pytest tests/unit/ -v --tb=short
        ;;
    2)
        echo "🔗 Running Integration Tests..."
        pytest tests/integration/ -v --tb=short
        ;;
    3)
        echo "⚡ Running Performance Tests..."
        pytest tests/performance/ -v --tb=short
        ;;
    4)
        echo "🎯 Running All Tests..."
        pytest tests/ -v --tb=short --cov=irst_library --cov-report=html --cov-report=term-missing
        echo ""
        echo "📊 Coverage report generated: htmlcov/index.html"
        ;;
    *)
        echo "❌ Invalid choice. Running all tests..."
        pytest tests/ -v --tb=short --cov=irst_library --cov-report=html --cov-report=term-missing
        ;;
esac

echo ""
echo "✅ Test execution completed!"
