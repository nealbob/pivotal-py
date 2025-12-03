import sys
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add parent directory to path so we can import pivotal
sys.path.append(PROJECT_ROOT)

from pivotal.dsl_parser import DSLParser

# Ensure data directory exists
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, 'data.csv')

# Create dummy data
with open(DATA_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'product', 'price', 'quantity', 'category'])
    writer.writerow([1, 'Laptop', 999.99, 5, 'Electronics'])
    writer.writerow([2, 'Mouse', 25.50, 150, 'Electronics'])
    writer.writerow([3, 'Desk', 299.00, 20, 'Furniture'])
    writer.writerow([4, 'Chair', 159.99, 45, 'Furniture'])
    writer.writerow([5, 'Monitor', 399.00, 30, 'Electronics'])
    writer.writerow([6, 'Keyboard', 79.99, 80, 'Electronics'])

parser = DSLParser()
PIVOTAL_FILE = os.path.join(SCRIPT_DIR, 'test_plot.pivotal')

with open(PIVOTAL_FILE, 'r') as f:
    code = f.read()

# Change working directory to examples so relative paths in DSL work
os.chdir(SCRIPT_DIR)

print("Parsing code...")
results = parser.parse(code)
if isinstance(results, dict) and 'error' in results:
    print(f"Error: {results['error']}")
    sys.exit(1)

print("Generating code...")
python_code = parser.generate_code(results)
for line in python_code:
    print(line)

print("\nExecuting code...")
# We need to pass globals() so the python block variables are available
# Mocking plt.show to avoid blocking
plt.show = lambda: print("Plot displayed")

tables = parser.execute(code, globals())

if tables:
    print("\nExecution successful!")
else:
    print("Execution failed or no tables returned.")

# Cleanup
if os.path.exists('data.csv'):
    os.remove('data.csv')
