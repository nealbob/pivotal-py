import sys
import os
# Add parent directory to path so we can import pivotal if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pivotal
import csv 
import importlib
importlib.reload(pivotal)

# Use path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'test.pivotal'), 'r') as f:
    dsl_code = f.read()

parser = pivotal.DSLParser()
results = parser.parse(dsl_code)

print("\n")
print("-" * 80)
print("Parse string to AST:")
print("-" * 80)
for res in results:
    print(res)

print("\n ")
print("-" * 80)
print("Parse to Python code (pandas):")
print("-" * 80)
code = parser.generate_code(results)
for c in code:
    print(c)

print("\n ")
print("-" * 80)
print("Parse file to Python (pandas):")
print("-" * 80)
for res in results:
    print(res)

print("-" * 80)
print("Execute code:")
print("-" * 80)
# Or execute directly:
tables = parser.execute(dsl_code, globals())

print("-" * 80)
print("Export code:")
print("-" * 80)
pycode = parser.export(dsl_code)
print(pycode)