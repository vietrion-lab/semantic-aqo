#!/bin/bash
# Master test runner - executes all test scripts in sequence

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "SENSE EMBEDDINGS SEARCH SYSTEM - TEST SUITE"
echo "============================================================"
echo ""

# Make all test scripts executable
chmod +x "$SCRIPT_DIR"/*.sh

# Array of test scripts in execution order
TESTS=(
    "03-verify-database.sh"
    "01-search-word.sh"
    "02-search-multiple-words.sh"
    "05-edge-cases.sh"
    "04-performance-benchmark.sh"
)

PASSED=0
FAILED=0

for test in "${TESTS[@]}"; do
    echo ""
    echo "▶ Running: $test"
    echo "------------------------------------------------------------"
    
    if bash "$SCRIPT_DIR/$test"; then
        echo "✅ PASSED: $test"
        ((PASSED++))
    else
        echo "❌ FAILED: $test"
        ((FAILED++))
    fi
    
    echo ""
    sleep 2
done

echo "============================================================"
echo "TEST SUITE SUMMARY"
echo "============================================================"
echo "Total tests:  $((PASSED + FAILED))"
echo "✅ Passed:    $PASSED"
echo "❌ Failed:    $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All tests passed successfully!"
    exit 0
else
    echo "⚠️  Some tests failed. Please review the output above."
    exit 1
fi
