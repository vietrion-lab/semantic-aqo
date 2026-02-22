#!/bin/bash
# Test 1: Search for a word that exists in the vocabulary

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "============================================================"
echo "TEST 1: Search for word 'ELSE'"
echo "============================================================"

python main.py search ELSE

echo ""
echo "✓ Test completed successfully"
