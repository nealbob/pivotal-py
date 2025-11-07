import dsl_parser 
import csv 
import os
import importlib
importlib.reload(dsl_parser)

with open('test.pivotal', 'r') as f:
    dsl_code = f.read()

parser = dsl_parser.DSLParser()
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