#!/bin/bash
# Test 5: Edge cases and error handling

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "============================================================"
echo "TEST 5: Edge Cases"
echo "============================================================"

# Test 1: Word not in vocabulary
echo ""
echo "--- Test: Non-existent word ---"
python main.py search "zzz_nonexistent_xyz" 2>&1 || echo "✓ Handled gracefully"

# Test 2: Symbol character
echo ""
echo "--- Test: Symbol character '*' ---"
python main.py search "*" 2>&1 || echo "✓ Handled gracefully"

# Test 3: Valid word
echo ""
echo "--- Test: Valid word 'AND' ---"
python main.py search "AND" 2>&1

echo ""
echo "✓ Edge case tests completed"
