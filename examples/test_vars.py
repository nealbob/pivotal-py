import sys
import os
import pandas as pd
import csv

# Add parent directory to path so we can import pivotal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pivotal.dsl_parser import DSLParser

# Create dummy data
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'product', 'price', 'quantity', 'category'])
    writer.writerow([1, 'Laptop', 999.99, 5, 'Electronics'])
    writer.writerow([2, 'Mouse', 25.50, 150, 'Electronics'])
    writer.writerow([3, 'Desk', 299.00, 20, 'Furniture'])
    writer.writerow([4, 'Chair', 159.99, 45, 'Furniture'])
    writer.writerow([5, 'Monitor', 399.00, 30, 'Electronics'])
    writer.writerow([6, 'Keyboard', 79.99, 80, 'Electronics'])

parser = DSLParser()
with open('examples/test_vars.pivotal', 'r') as f:
    code = f.read()

print("Parsing code...")
results = parser.parse(code)
if isinstance(results, dict) and 'error' in results:
    print(f"Error: {results['error']}")
    sys.exit(1)

print("Executing code...")
# We need to pass globals() so the python block variables are available
tables = parser.execute(code, globals())

if tables:
    print("\nExecution successful!")
    for name, df in tables.items():
        print(f"\nTable '{name}':")
        print(df)
else:
    print("Execution failed or no tables returned.")

# Cleanup
if os.path.exists('data.csv'):
    os.remove('data.csv')
