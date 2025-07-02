#!/usr/bin/env python3
"""Generate comprehensive documentation from docstrings and code analysis."""

import ast
import inspect
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Any
import json


def extract_class_info(cls) -> Dict[str, Any]:
    """Extract comprehensive information about a class."""
    info = {
        'name': cls.__name__,
        'docstring': inspect.getdoc(cls) or '',
        'methods': [],
        'properties': [],
        'module': cls.__module__,
    }
    
    for name, method in inspect.getmembers(cls, inspect.ismethod):
        if not name.startswith('_'):
            method_info = {
                'name': name,
                'docstring': inspect.getdoc(method) or '',
                'signature': str(inspect.signature(method)) if hasattr(inspect, 'signature') else '',
            }
            info['methods'].append(method_info)
    
    for name, prop in inspect.getmembers(cls, lambda x: isinstance(x, property)):
        prop_info = {
            'name': name,
            'docstring': inspect.getdoc(prop) or '',
        }
        info['properties'].append(prop_info)
    
    return info


def extract_function_info(func) -> Dict[str, Any]:
    """Extract comprehensive information about a function."""
    return {
        'name': func.__name__,
        'docstring': inspect.getdoc(func) or '',
        'signature': str(inspect.signature(func)) if hasattr(inspect, 'signature') else '',
        'module': func.__module__,
    }


def scan_package(package_name: str) -> Dict[str, Any]:
    """Scan a package and extract all documentation."""
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        print(f"Could not import {package_name}: {e}")
        return {}
    
    documentation = {
        'classes': [],
        'functions': [],
        'modules': []
    }
    
    # Get package path
    package_path = Path(package.__file__).parent
    
    # Scan all Python files in the package
    for py_file in package_path.rglob('*.py'):
        if py_file.name.startswith('_'):
            continue
        
        # Convert file path to module name
        relative_path = py_file.relative_to(package_path.parent)
        module_name = str(relative_path.with_suffix('')).replace('/', '.').replace('\\', '.')
        
        try:
            module = importlib.import_module(module_name)
            
            # Extract classes
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module_name:
                    documentation['classes'].append(extract_class_info(cls))
            
            # Extract functions
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if func.__module__ == module_name and not name.startswith('_'):
                    documentation['functions'].append(extract_function_info(func))
        
        except Exception as e:
            print(f"Error processing {module_name}: {e}")
    
    return documentation


def generate_markdown_docs(documentation: Dict[str, Any], output_path: Path):
    """Generate markdown documentation from extracted information."""
    content = ["# API Documentation", ""]
    
    # Classes section
    if documentation['classes']:
        content.extend(["## Classes", ""])
        for cls_info in documentation['classes']:
            content.append(f"### {cls_info['name']}")
            content.append("")
            if cls_info['docstring']:
                content.append(cls_info['docstring'])
                content.append("")
            
            content.append(f"**Module:** `{cls_info['module']}`")
            content.append("")
            
            if cls_info['methods']:
                content.append("#### Methods")
                content.append("")
                for method in cls_info['methods']:
                    content.append(f"##### `{method['name']}{method['signature']}`")
                    content.append("")
                    if method['docstring']:
                        content.append(method['docstring'])
                        content.append("")
            
            if cls_info['properties']:
                content.append("#### Properties")
                content.append("")
                for prop in cls_info['properties']:
                    content.append(f"##### `{prop['name']}`")
                    content.append("")
                    if prop['docstring']:
                        content.append(prop['docstring'])
                        content.append("")
            
            content.append("---")
            content.append("")
    
    # Functions section
    if documentation['functions']:
        content.extend(["## Functions", ""])
        for func_info in documentation['functions']:
            content.append(f"### `{func_info['name']}{func_info['signature']}`")
            content.append("")
            if func_info['docstring']:
                content.append(func_info['docstring'])
                content.append("")
            content.append(f"**Module:** `{func_info['module']}`")
            content.append("")
            content.append("---")
            content.append("")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(content))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate API documentation')
    parser.add_argument('package', help='Package name to document')
    parser.add_argument('--output', default='docs/api_generated.md', help='Output file path')
    parser.add_argument('--json', help='Also output JSON format to this path')
    
    args = parser.parse_args()
    
    print(f"Scanning package: {args.package}")
    documentation = scan_package(args.package)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating markdown documentation: {output_path}")
    generate_markdown_docs(documentation, output_path)
    
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Generating JSON documentation: {json_path}")
        with open(json_path, 'w') as f:
            json.dump(documentation, f, indent=2)
    
    print("Documentation generation complete!")


if __name__ == '__main__':
    main()
