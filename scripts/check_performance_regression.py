#!/usr/bin/env python3
"""Performance regression detection script."""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_benchmark_data(filepath: Path) -> Dict[str, Any]:
    """Load benchmark data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_regression(current: float, baseline: float, threshold: float = 0.1) -> bool:
    """Check if there's a performance regression."""
    if baseline == 0:
        return False
    
    regression_ratio = (current - baseline) / baseline
    return regression_ratio > threshold


def analyze_benchmarks(current_data: Dict, baseline_data: Dict) -> List[Dict]:
    """Analyze benchmark data for regressions."""
    regressions = []
    
    for benchmark_name, current_stats in current_data.get('benchmarks', {}).items():
        if benchmark_name not in baseline_data.get('benchmarks', {}):
            continue
        
        baseline_stats = baseline_data['benchmarks'][benchmark_name]
        current_mean = current_stats.get('stats', {}).get('mean', 0)
        baseline_mean = baseline_stats.get('stats', {}).get('mean', 0)
        
        if calculate_regression(current_mean, baseline_mean):
            regression_percent = ((current_mean - baseline_mean) / baseline_mean) * 100
            regressions.append({
                'benchmark': benchmark_name,
                'current_mean': current_mean,
                'baseline_mean': baseline_mean,
                'regression_percent': regression_percent
            })
    
    return regressions


def main():
    parser = argparse.ArgumentParser(description='Check for performance regressions')
    parser.add_argument('--current', required=True, help='Current benchmark JSON file')
    parser.add_argument('--baseline', required=True, help='Baseline benchmark JSON file')
    parser.add_argument('--threshold', type=float, default=0.1, help='Regression threshold (default: 0.1 = 10%)')
    
    args = parser.parse_args()
    
    current_path = Path(args.current)
    baseline_path = Path(args.baseline)
    
    if not current_path.exists():
        print(f"Current benchmark file not found: {current_path}")
        return 0
    
    if not baseline_path.exists():
        print(f"Baseline benchmark file not found: {baseline_path}")
        print("Creating baseline from current results...")
        current_data = load_benchmark_data(current_path)
        with open(baseline_path, 'w') as f:
            json.dump(current_data, f, indent=2)
        return 0
    
    current_data = load_benchmark_data(current_path)
    baseline_data = load_benchmark_data(baseline_path)
    
    regressions = analyze_benchmarks(current_data, baseline_data)
    
    if regressions:
        print("⚠️  Performance regressions detected:")
        for regression in regressions:
            print(f"  {regression['benchmark']}: {regression['regression_percent']:.1f}% slower")
            print(f"    Current: {regression['current_mean']:.4f}s")
            print(f"    Baseline: {regression['baseline_mean']:.4f}s")
        sys.exit(1)
    else:
        print("✅ No performance regressions detected")
        return 0


if __name__ == '__main__':
    main()
