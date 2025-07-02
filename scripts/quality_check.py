#!/usr/bin/env python3
"""
Professional code quality checker for IRST Library.
Runs comprehensive code quality checks including static analysis, security, and style.
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class QualityCheck:
    """Represents a single quality check result."""
    name: str
    passed: bool
    score: Optional[float] = None
    issues: List[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.details is None:
            self.details = {}


class CodeQualityChecker:
    """Comprehensive code quality checking tool."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.results: List[QualityCheck] = []
    
    def run_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command safely."""
        try:
            result = subprocess.run(
                command,
                cwd=self.project_path,
                capture_output=capture_output,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(command, 1, "", "Command timed out")
        except Exception as e:
            return subprocess.CompletedProcess(command, 1, "", str(e))
    
    def check_flake8_style(self) -> QualityCheck:
        """Check PEP8 compliance with flake8."""
        result = self.run_command(['flake8', '--count', '--statistics', 'irst_library/'])
        
        if result.returncode == 0:
            return QualityCheck(
                name="flake8_style",
                passed=True,
                score=100.0,
                details={"output": result.stdout}
            )
        else:
            issues = result.stdout.strip().split('\n') if result.stdout else ["No specific issues found"]
            return QualityCheck(
                name="flake8_style",
                passed=False,
                score=0.0,
                issues=issues,
                details={"stderr": result.stderr}
            )
    
    def check_black_formatting(self) -> QualityCheck:
        """Check code formatting with black."""
        result = self.run_command(['black', '--check', '--diff', 'irst_library/'])
        
        if result.returncode == 0:
            return QualityCheck(
                name="black_formatting",
                passed=True,
                score=100.0
            )
        else:
            return QualityCheck(
                name="black_formatting",
                passed=False,
                score=0.0,
                issues=["Code formatting issues found"],
                details={"diff": result.stdout}
            )
    
    def check_isort_imports(self) -> QualityCheck:
        """Check import sorting with isort."""
        result = self.run_command(['isort', '--check-only', '--diff', 'irst_library/'])
        
        if result.returncode == 0:
            return QualityCheck(
                name="isort_imports",
                passed=True,
                score=100.0
            )
        else:
            return QualityCheck(
                name="isort_imports",
                passed=False,
                score=0.0,
                issues=["Import sorting issues found"],
                details={"diff": result.stdout}
            )
    
    def check_mypy_types(self) -> QualityCheck:
        """Check type hints with mypy."""
        result = self.run_command(['mypy', 'irst_library/'])
        
        if result.returncode == 0:
            return QualityCheck(
                name="mypy_types",
                passed=True,
                score=100.0
            )
        else:
            issues = result.stdout.strip().split('\n') if result.stdout else []
            return QualityCheck(
                name="mypy_types",
                passed=False,
                score=max(0, 100 - len(issues) * 5),  # Deduct 5 points per issue
                issues=issues[:10],  # Show only first 10 issues
                details={"total_issues": len(issues)}
            )
    
    def check_bandit_security(self) -> QualityCheck:
        """Check security issues with bandit."""
        result = self.run_command(['bandit', '-r', 'irst_library/', '-f', 'json'])
        
        try:
            if result.returncode == 0:
                return QualityCheck(
                    name="bandit_security",
                    passed=True,
                    score=100.0
                )
            else:
                # Parse JSON output
                bandit_data = json.loads(result.stdout) if result.stdout else {}
                issues = bandit_data.get('results', [])
                
                # Calculate score based on severity
                high_severity = len([i for i in issues if i.get('issue_severity') == 'HIGH'])
                medium_severity = len([i for i in issues if i.get('issue_severity') == 'MEDIUM'])
                low_severity = len([i for i in issues if i.get('issue_severity') == 'LOW'])
                
                score = max(0, 100 - (high_severity * 20 + medium_severity * 10 + low_severity * 5))
                
                return QualityCheck(
                    name="bandit_security",
                    passed=len(issues) == 0,
                    score=score,
                    issues=[f"{i['test_name']}: {i['issue_text']}" for i in issues[:5]],
                    details={
                        "high_severity": high_severity,
                        "medium_severity": medium_severity,
                        "low_severity": low_severity,
                        "total_issues": len(issues)
                    }
                )
        except json.JSONDecodeError:
            return QualityCheck(
                name="bandit_security",
                passed=False,
                score=0.0,
                issues=["Failed to parse bandit output"],
                details={"raw_output": result.stdout}
            )
    
    def check_pylint_quality(self) -> QualityCheck:
        """Check code quality with pylint."""
        result = self.run_command(['pylint', '--output-format=json', 'irst_library/'])
        
        try:
            if result.stdout:
                pylint_data = json.loads(result.stdout)
                
                # Count issues by type
                errors = len([m for m in pylint_data if m['type'] == 'error'])
                warnings = len([m for m in pylint_data if m['type'] == 'warning'])
                refactors = len([m for m in pylint_data if m['type'] == 'refactor'])
                conventions = len([m for m in pylint_data if m['type'] == 'convention'])
                
                total_issues = errors + warnings + refactors + conventions
                score = max(0, 100 - (errors * 10 + warnings * 5 + refactors * 2 + conventions * 1))
                
                return QualityCheck(
                    name="pylint_quality",
                    passed=total_issues == 0,
                    score=score,
                    issues=[f"{m['message']} ({m['symbol']})" for m in pylint_data[:5]],
                    details={
                        "errors": errors,
                        "warnings": warnings,
                        "refactors": refactors,
                        "conventions": conventions,
                        "total_issues": total_issues
                    }
                )
            else:
                return QualityCheck(
                    name="pylint_quality",
                    passed=True,
                    score=100.0
                )
        except json.JSONDecodeError:
            return QualityCheck(
                name="pylint_quality",
                passed=False,
                score=0.0,
                issues=["Failed to parse pylint output"]
            )
    
    def check_test_coverage(self) -> QualityCheck:
        """Check test coverage."""
        result = self.run_command(['coverage', 'run', '-m', 'pytest'])
        if result.returncode != 0:
            return QualityCheck(
                name="test_coverage",
                passed=False,
                score=0.0,
                issues=["Tests failed to run"],
                details={"error": result.stderr}
            )
        
        # Get coverage report
        result = self.run_command(['coverage', 'report', '--format=json'])
        try:
            if result.stdout:
                coverage_data = json.loads(result.stdout)
                coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
                
                return QualityCheck(
                    name="test_coverage",
                    passed=coverage_percent >= 80,
                    score=coverage_percent,
                    details={"coverage_percent": coverage_percent}
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback to text output
        result = self.run_command(['coverage', 'report'])
        lines = result.stdout.strip().split('\n') if result.stdout else []
        
        # Look for total coverage line
        for line in lines:
            if 'TOTAL' in line:
                parts = line.split()
                if len(parts) >= 4 and '%' in parts[-1]:
                    coverage_percent = float(parts[-1].replace('%', ''))
                    return QualityCheck(
                        name="test_coverage",
                        passed=coverage_percent >= 80,
                        score=coverage_percent,
                        details={"coverage_percent": coverage_percent}
                    )
        
        return QualityCheck(
            name="test_coverage",
            passed=False,
            score=0.0,
            issues=["Could not determine coverage"]
        )
    
    def check_documentation(self) -> QualityCheck:
        """Check documentation completeness."""
        doc_files = [
            'README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE',
            'docs/api_reference.md', 'docs/quickstart.md'
        ]
        
        missing_docs = []
        for doc_file in doc_files:
            if not (self.project_path / doc_file).exists():
                missing_docs.append(doc_file)
        
        score = max(0, 100 - len(missing_docs) * 15)
        
        return QualityCheck(
            name="documentation",
            passed=len(missing_docs) == 0,
            score=score,
            issues=[f"Missing: {doc}" for doc in missing_docs],
            details={"required_docs": doc_files, "missing_docs": missing_docs}
        )
    
    def run_all_checks(self) -> List[QualityCheck]:
        """Run all quality checks."""
        checks = [
            self.check_flake8_style,
            self.check_black_formatting,
            self.check_isort_imports,
            self.check_mypy_types,
            self.check_bandit_security,
            self.check_pylint_quality,
            self.check_test_coverage,
            self.check_documentation,
        ]
        
        self.results = []
        for check_func in checks:
            print(f"Running {check_func.__name__}...")
            try:
                result = check_func()
                self.results.append(result)
            except Exception as e:
                self.results.append(
                    QualityCheck(
                        name=check_func.__name__,
                        passed=False,
                        score=0.0,
                        issues=[f"Check failed: {str(e)}"]
                    )
                )
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        average_score = sum(r.score or 0 for r in self.results if r.score is not None) / total_checks
        
        return {
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks,
                "pass_rate": (passed_checks / total_checks) * 100,
                "average_score": average_score,
                "overall_grade": self._calculate_grade(average_score)
            },
            "checks": [asdict(result) for result in self.results],
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 95: return "A+"
        elif score >= 90: return "A"
        elif score >= 85: return "A-"
        elif score >= 80: return "B+"
        elif score >= 75: return "B"
        elif score >= 70: return "B-"
        elif score >= 65: return "C+"
        elif score >= 60: return "C"
        elif score >= 55: return "C-"
        elif score >= 50: return "D"
        else: return "F"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for result in self.results:
            if not result.passed:
                if result.name == "flake8_style":
                    recommendations.append("Run 'flake8 --max-line-length=88' to identify style issues")
                elif result.name == "black_formatting":
                    recommendations.append("Run 'black irst_library/' to auto-format code")
                elif result.name == "isort_imports":
                    recommendations.append("Run 'isort irst_library/' to sort imports")
                elif result.name == "mypy_types":
                    recommendations.append("Add type hints to improve code clarity and catch errors")
                elif result.name == "bandit_security":
                    recommendations.append("Review security issues identified by bandit")
                elif result.name == "test_coverage":
                    recommendations.append("Add more unit tests to increase coverage above 80%")
                elif result.name == "documentation":
                    recommendations.append("Create missing documentation files")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Code Quality Checker')
    parser.add_argument('--project-path', type=Path, default=Path('.'),
                       help='Path to project directory')
    parser.add_argument('--output', type=Path, help='Output JSON report file')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')
    
    args = parser.parse_args()
    
    checker = CodeQualityChecker(args.project_path)
    results = checker.run_all_checks()
    report = checker.generate_report()
    
    if args.format == 'json':
        output = json.dumps(report, indent=2)
        print(output)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
    else:
        # Text format
        print("\n" + "="*80)
        print("CODE QUALITY REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"Overall Grade: {summary['overall_grade']} ({summary['average_score']:.1f}/100)")
        print(f"Checks Passed: {summary['passed_checks']}/{summary['total_checks']} ({summary['pass_rate']:.1f}%)")
        print()
        
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            score_str = f"({result.score:.1f}/100)" if result.score is not None else ""
            print(f"{result.name:<20}: {status} {score_str}")
            
            if result.issues:
                for issue in result.issues[:3]:  # Show first 3 issues
                    print(f"  • {issue}")
                if len(result.issues) > 3:
                    print(f"  • ... and {len(result.issues) - 3} more")
        
        if report['recommendations']:
            print("\n" + "-"*80)
            print("RECOMMENDATIONS")
            print("-"*80)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
    
    # Exit with error code if quality is poor
    if report['summary']['average_score'] < 70:
        sys.exit(1)


if __name__ == "__main__":
    main()
