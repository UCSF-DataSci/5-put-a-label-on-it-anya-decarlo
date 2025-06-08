#!/usr/bin/env python
import nbformat as nbf
import re
import os

def create_part1_notebook():
    """Create part1_introduction.ipynb notebook from part1_functions.py"""
    # Read the Python file
    with open('part1_functions.py', 'r') as file:
        content = file.read()
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add markdown cell with introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# Part 1: Introduction to Classification
    
This notebook implements a basic classification model for health data.
    """))
    
    # Add imports cell
    import_pattern = r'import.*?\n\n'
    imports = re.search(import_pattern, content, re.DOTALL)
    if imports:
        nb.cells.append(nbf.v4.new_code_cell(imports.group(0)))
    
    # Find and add each function as a separate cell
    function_pattern = r'def\s+(\w+)\s*\([^)]*\):\s*""".*?""".*?(?=\n\ndef|\n\n#|$)'
    functions = re.finditer(function_pattern, content, re.DOTALL)
    
    for func_match in functions:
        func_name = func_match.group(1)
        func_code = func_match.group(0)
        
        # Add markdown cell describing the function
        nb.cells.append(nbf.v4.new_markdown_cell(f"## Function: {func_name}"))
        
        # Add the function code
        nb.cells.append(nbf.v4.new_code_cell(func_code))
    
    # Add main execution
    main_pattern = r'if\s+__name__\s*==\s*"__main__".*'
    main = re.search(main_pattern, content, re.DOTALL)
    if main:
        nb.cells.append(nbf.v4.new_markdown_cell("## Main Execution"))
        nb.cells.append(nbf.v4.new_code_cell(main.group(0)))
    
    # Save the notebook
    with open('part1_introduction.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Created part1_introduction.ipynb")

def create_part2_notebook():
    """Create part2_feature_engineering.ipynb notebook from part2_functions.py"""
    # Read the Python file
    with open('part2_functions.py', 'r') as file:
        content = file.read()
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add markdown cell with introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# Part 2: Feature Engineering for Time-Series Health Data
    
This notebook implements time-series feature engineering and advanced tree-based models.
    """))
    
    # Add imports cell
    import_pattern = r'import.*?\n\n'
    imports = re.search(import_pattern, content, re.DOTALL)
    if imports:
        nb.cells.append(nbf.v4.new_code_cell(imports.group(0)))
    
    # Find and add each function as a separate cell
    function_pattern = r'def\s+(\w+)\s*\([^)]*\):\s*""".*?""".*?(?=\n\ndef|\n\n#|$)'
    functions = re.finditer(function_pattern, content, re.DOTALL)
    
    for func_match in functions:
        func_name = func_match.group(1)
        func_code = func_match.group(0)
        
        # Add markdown cell describing the function
        nb.cells.append(nbf.v4.new_markdown_cell(f"## Function: {func_name}"))
        
        # Add the function code
        nb.cells.append(nbf.v4.new_code_cell(func_code))
    
    # Add main execution
    main_pattern = r'if\s+__name__\s*==\s*"__main__".*'
    main = re.search(main_pattern, content, re.DOTALL)
    if main:
        nb.cells.append(nbf.v4.new_markdown_cell("## Main Execution"))
        nb.cells.append(nbf.v4.new_code_cell(main.group(0)))
    
    # Save the notebook
    with open('part2_feature_engineering.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Created part2_feature_engineering.ipynb")

def create_part3_notebook():
    """Create part3_data_preparation.ipynb notebook from part3_functions.py"""
    # Read the Python file
    with open('part3_functions.py', 'r') as file:
        content = file.read()
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add markdown cell with introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# Part 3: Practical Data Preparation
    
This notebook implements categorical feature encoding with One-Hot Encoding and class imbalance handling with SMOTE.
    """))
    
    # Add imports cell
    import_pattern = r'import.*?\n\n'
    imports = re.search(import_pattern, content, re.DOTALL)
    if imports:
        nb.cells.append(nbf.v4.new_code_cell(imports.group(0)))
    
    # Find and add each function as a separate cell
    function_pattern = r'def\s+(\w+)\s*\([^)]*\):\s*""".*?""".*?(?=\n\ndef|\n\n#|$)'
    functions = re.finditer(function_pattern, content, re.DOTALL)
    
    for func_match in functions:
        func_name = func_match.group(1)
        func_code = func_match.group(0)
        
        # Add markdown cell describing the function
        nb.cells.append(nbf.v4.new_markdown_cell(f"## Function: {func_name}"))
        
        # Add the function code
        nb.cells.append(nbf.v4.new_code_cell(func_code))
    
    # Add main execution
    main_pattern = r'if\s+__name__\s*==\s*"__main__".*'
    main = re.search(main_pattern, content, re.DOTALL)
    if main:
        nb.cells.append(nbf.v4.new_markdown_cell("## Main Execution"))
        nb.cells.append(nbf.v4.new_code_cell(main.group(0)))
    
    # Save the notebook
    with open('part3_data_preparation.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Created part3_data_preparation.ipynb")

if __name__ == "__main__":
    # Create notebooks
    create_part1_notebook()
    create_part2_notebook() 
    create_part3_notebook()
    print("All notebooks created successfully!")
