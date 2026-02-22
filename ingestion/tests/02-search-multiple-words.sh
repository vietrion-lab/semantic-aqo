#!/bin/bash
# Test 2: Search for multiple words from the vocabulary

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

WORDS=("SELECT" "FROM" "WHERE" "JOIN" "DISTINCT")

echo "============================================================"
echo "TEST 2: Search for multiple words"
echo "============================================================"

for word in "${WORDS[@]}"; do
    echo ""
    echo "--- Searching for: $word ---"
    python main.py search "$word"
done

echo ""
echo "✓ Test completed successfully"
