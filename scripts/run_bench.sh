#!/bin/bash
# SPDX-FileCopyrightText: 2026 RVSPOC Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Run benchmark against both RVV and scalar library builds.
# Produces a side-by-side comparison with speedup ratios.
#
# Prerequisites:
#   - build/      directory with RVV build (cmake -DCMAKE_BUILD_TYPE=Release)
#   - build_scalar/ directory with scalar build (same but -march=rv64gc)
#
# Usage: cd rvspoc-P2601-kleidicv && bash scripts/run_bench.sh

set -e
cd "$(dirname "$0")/.."

BENCH_SRC=scripts/bench.cpp
INCLUDE="-Ikleidicv/include"

echo "=== Building RVV benchmark ==="
c++ -O2 -std=c++17 $INCLUDE -Ibuild/kleidicv/include \
    -o /tmp/bench_rvv "$BENCH_SRC" -Lbuild/kleidicv -lkleidicv 2>&1 | grep -v warning || true

echo "=== Building Scalar benchmark ==="
c++ -O2 -std=c++17 -march=rv64gc $INCLUDE -Ibuild_scalar/kleidicv/include \
    -o /tmp/bench_scalar "$BENCH_SRC" -Lbuild_scalar/kleidicv -lkleidicv 2>&1 | grep -v warning || true

echo ""
echo "=== Running RVV benchmark ==="
/tmp/bench_rvv > /tmp/results_rvv.txt
cat /tmp/results_rvv.txt

echo ""
echo "=== Running Scalar benchmark ==="
/tmp/bench_scalar > /tmp/results_scalar.txt
cat /tmp/results_scalar.txt

echo ""
echo "============================================"
echo "=== Speedup Comparison (Scalar vs RVV) ==="
echo "============================================"
printf "%-35s %10s %10s %8s\n" "Operation" "Scalar" "RVV" "Speedup"
printf "%-35s %10s %10s %8s\n" "---------" "------" "---" "-------"

paste /tmp/results_scalar.txt /tmp/results_rvv.txt | while IFS=$'\t' read -r sline rline; do
    sname=$(echo "$sline" | awk '{print $1}')
    sms=$(echo "$sline" | awk '{print $2}')
    rms=$(echo "$rline" | awk '{print $2}')

    # Skip header/section lines
    if echo "$sname" | grep -qE '^(KleidiCV|===|$)'; then
        echo "$sline"
        continue
    fi

    if [ -n "$sms" ] && [ -n "$rms" ] && [ "$rms" != "0.0000" ]; then
        speedup=$(awk "BEGIN{printf \"%.1f\", $sms/$rms}")
        printf "%-35s %8s ms %8s ms %7sx\n" "$sname" "$sms" "$rms" "$speedup"
    fi
done
