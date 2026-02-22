#!/bin/bash
# Test 4: Performance benchmark for PQ search

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "============================================================"
echo "TEST 4: Performance Benchmark"
echo "============================================================"

TEST_WORDS=("SELECT" "FROM" "WHERE" "AND" "OR" "ELSE" "CASE" "COUNT" "ORDER" "BETWEEN")

total_time=0
num_tests=${#TEST_WORDS[@]}

echo "Running $num_tests search queries..."
echo ""

for word in "${TEST_WORDS[@]}"; do
    echo -n "Searching '$word'... "
    start=$(date +%s%N)
    python main.py search "$word" > /dev/null 2>&1
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))  # milliseconds
    total_time=$((total_time + elapsed))
    echo "✓ ${elapsed}ms"
done

avg_time=$((total_time / num_tests))

echo ""
echo "============================================================"
echo "BENCHMARK RESULTS"
echo "============================================================"
echo "Total queries:    $num_tests"
echo "Total time:       ${total_time}ms"
echo "Average time:     ${avg_time}ms per query"
echo ""
echo "✓ Performance benchmark completed"
