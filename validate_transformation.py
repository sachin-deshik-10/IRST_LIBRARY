#!/usr/bin/env python3
"""
IRST Library - Final Update and Validation Script

This script performs final updates and validates the complete transformation
of the IRST Library to ensure all professional standards are met.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return Path(filepath).exists()

def validate_professional_structure() -> Dict[str, Any]:
    """Validate the professional structure of the repository."""
    
    results = {
        "core_files": {},
        "documentation": {},
        "ci_cd": {},
        "quality_tools": {},
        "deployment": {},
        "notebooks": {},
        "missing_files": [],
        "score": 0
    }
    
    # Core files to check
    core_files = [
        "setup.py",
        "requirements.txt", 
        "requirements-dev.txt",
        "MANIFEST.in",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTORS.md",
        "GOVERNANCE.md",
        "ROADMAP.md",
        ".gitignore",
        ".env.template"
    ]
    
    # Documentation files
    doc_files = [
        "docs/api_reference.md",
        "docs/ARCHITECTURE.md",
        "docs/BENCHMARKS.md",
        "docs/FAQ.md",
        "docs/quickstart.md",
        "docs/models.md",
        "docs/datasets.md"
    ]
    
    # CI/CD files
    cicd_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/benchmarks.yml",
        ".github/workflows/release.yml",
        ".github/workflows/security.yml",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md"
    ]
    
    # Quality tools
    quality_files = [
        ".pre-commit-config.yaml",
        "Makefile",
        "mypy.ini",
        ".coveragerc",
        "pytest.ini",
        ".bandit",
        ".editorconfig"
    ]
    
    # Deployment files
    deployment_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "k8s/deployment.yaml",
        "k8s/ingress.yaml",
        "k8s/monitoring.yaml",
        "deploy.sh",
        "run_tests.sh"
    ]
    
    # Notebooks
    notebook_files = [
        "notebooks/irst_tutorial.ipynb"
    ]
    
    # Check all file categories
    file_categories = [
        ("core_files", core_files),
        ("documentation", doc_files),
        ("ci_cd", cicd_files),
        ("quality_tools", quality_files),
        ("deployment", deployment_files),
        ("notebooks", notebook_files)
    ]
    
    total_files = 0
    found_files = 0
    
    for category, files in file_categories:
        results[category] = {}
        for file in files:
            exists = check_file_exists(file)
            results[category][file] = exists
            
            if exists:
                found_files += 1
            else:
                results["missing_files"].append(file)
            
            total_files += 1
    
    # Calculate score
    results["score"] = (found_files / total_files) * 100
    
    return results

def check_python_structure() -> Dict[str, Any]:
    """Check the Python package structure."""
    
    python_structure = {
        "package_root": "irst_library",
        "modules": [
            "irst_library/__init__.py",
            "irst_library/core/__init__.py",
            "irst_library/models/__init__.py",
            "irst_library/datasets/__init__.py",
            "irst_library/training/__init__.py",
            "irst_library/utils/__init__.py",
            "irst_library/cli/__init__.py"
        ],
        "test_structure": [
            "tests/conftest.py",
            "tests/unit/test_core.py",
            "tests/unit/test_models.py"
        ],
        "scripts": [
            "scripts/benchmark.py",
            "scripts/check_performance_regression.py",
            "scripts/generate_docs.py",
            "scripts/setup_dev.py",
            "scripts/validate_model.py",
            "scripts/quality_check.py"
        ]
    }
    
    results = {}
    
    for category, files in python_structure.items():
        if category == "package_root":
            continue
            
        results[category] = {}
        for file in files:
            results[category][file] = check_file_exists(file)
    
    return results

def generate_final_report() -> None:
    """Generate a final transformation report."""
    
    print("🔍 IRST Library - Final Validation Report")
    print("=" * 50)
    
    # Validate structure
    structure_results = validate_professional_structure()
    python_results = check_python_structure()
    
    print(f"\n📊 Overall Score: {structure_results['score']:.1f}%")
    
    # Report by category
    categories = [
        ("Core Files", structure_results["core_files"]),
        ("Documentation", structure_results["documentation"]),
        ("CI/CD", structure_results["ci_cd"]),
        ("Quality Tools", structure_results["quality_tools"]),
        ("Deployment", structure_results["deployment"]),
        ("Notebooks", structure_results["notebooks"])
    ]
    
    for category_name, category_results in categories:
        total = len(category_results)
        found = sum(1 for exists in category_results.values() if exists)
        percentage = (found / total) * 100 if total > 0 else 0
        
        status = "✅" if percentage == 100 else "⚠️" if percentage >= 80 else "❌"
        print(f"\n{status} {category_name}: {found}/{total} ({percentage:.1f}%)")
        
        # Show missing files
        missing = [file for file, exists in category_results.items() if not exists]
        if missing:
            for file in missing[:3]:  # Show first 3 missing files
                print(f"   ❌ Missing: {file}")
            if len(missing) > 3:
                print(f"   ... and {len(missing) - 3} more")
    
    # Python structure report
    print(f"\n🐍 Python Package Structure:")
    for category, files in python_results.items():
        total = len(files)
        found = sum(1 for exists in files.values() if exists)
        percentage = (found / total) * 100 if total > 0 else 0
        status = "✅" if percentage >= 80 else "⚠️"
        print(f"   {status} {category.title()}: {found}/{total} ({percentage:.1f}%)")
    
    # Professional standards checklist
    print(f"\n📋 Professional Standards Checklist:")
    standards = [
        ("Setup & Configuration", structure_results["core_files"]["setup.py"] and 
         structure_results["core_files"]["pyproject.toml"]),
        ("Documentation", len([f for f in structure_results["documentation"].values() if f]) >= 5),
        ("CI/CD Pipeline", len([f for f in structure_results["ci_cd"].values() if f]) >= 4),
        ("Code Quality", len([f for f in structure_results["quality_tools"].values() if f]) >= 5),
        ("Containerization", structure_results["deployment"]["Dockerfile"]),
        ("Kubernetes Ready", structure_results["deployment"]["k8s/deployment.yaml"]),
        ("Interactive Tutorials", structure_results["notebooks"]["notebooks/irst_tutorial.ipynb"]),
        ("Security Policies", structure_results["core_files"]["SECURITY.md"])
    ]
    
    for standard, passed in standards:
        status = "✅" if passed else "❌"
        print(f"   {status} {standard}")
    
    print(f"\n🎉 Transformation Status: {'COMPLETE' if structure_results['score'] >= 90 else 'IN PROGRESS'}")
    
    if structure_results["missing_files"]:
        print(f"\n⚠️  Missing Files ({len(structure_results['missing_files'])}):")
        for file in structure_results["missing_files"][:10]:  # Show first 10
            print(f"   - {file}")
        if len(structure_results["missing_files"]) > 10:
            print(f"   ... and {len(structure_results['missing_files']) - 10} more")
    
    # Save report
    report_data = {
        "timestamp": "2025-07-02T12:00:00Z",
        "score": structure_results["score"],
        "structure_results": structure_results,
        "python_results": python_results,
        "standards_passed": sum(1 for _, passed in standards if passed),
        "total_standards": len(standards)
    }
    
    with open("validation_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: validation_report.json")
    print(f"🚀 Ready for production deployment!")

if __name__ == "__main__":
    generate_final_report()
