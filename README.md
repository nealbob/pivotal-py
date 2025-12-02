# Pivotal

<img src="pivotal_logo.svg" width="120">

**Pivotal** is a Python-based Domain-Specific Language (DSL) for data processing and transformation. It provides a clean, readable syntax for common data operations, compiling to pandas code under the hood.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Language Syntax](#language-syntax)
  - [Loading Data](#loading-data)
  - [Table Operations](#table-operations)
  - [Filtering](#filtering)
  - [Selecting Columns](#selecting-columns)
  - [Creating/Modifying Columns](#creatingmodifying-columns)
  - [Sorting](#sorting)
  - [Merging Tables](#merging-tables)
  - [Pivot Tables](#pivot-tables)
- [VS Code Extension](#vs-code-extension)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Features

✨ **Clean, Readable Syntax** - Write data transformations in an intuitive, English-like language  
🐼 **Pandas-Powered** - Compiles to efficient pandas operations  
🔄 **Pipeline-Oriented** - Chain operations naturally with indentation-based blocks  
📊 **Rich Operations** - Load, filter, select, sort, merge, pivot, and transform data  
🎯 **Type-Aware** - Intelligent handling of different data types  
🔍 **Interactive** - Execute code directly in Python REPL with instant feedback  
🚀 **VS Code Integration** - Syntax highlighting and intelligent autocomplete

---

## Installation

### Prerequisites
- Python 3.7+
- pandas
- lark-parser

### Install Dependencies

```bash
pip install pandas lark-parser
```

### Setup

1. Clone or download the Pivotal DSL files
2. Install the package:
   ```bash
   pip install .
   ```
3. (Optional) Install the [VS Code extension](#vs-code-extension) for enhanced editing

---

## Quick Start

### Basic Usage

```python
import pivotal

# Create parser instance
parser = pivotal.DSLParser()

# Write Pivotal DSL code
dsl_code = """
load sales sales_data.csv

table high_value_sales:
    filter amount > 1000
    select customer_id, product, amount
    sort amount desc
"""

# Execute the DSL code
tables = parser.execute(dsl_code, globals())

# Access the resulting DataFrame
print(high_value_sales)
```

### Run from File

```python
# Read from .pivotal file
with open('analysis.pivotal', 'r') as f:
    dsl_code = f.read()

parser = pivotal.DSLParser()
parser.execute(dsl_code, globals())
```

---

## Language Syntax

### Loading Data

Load CSV files into named tables:

```pivotal
# Basic load
load sales data.csv

# Load then select 
load customers users.csv
   select name, email, city
   
# Load with custom names
load inventory "inventory_2024.csv"
   names ["product", "quantity", "price"]
   header 0
```

**Parameters:**
- Accepts all keyword arguments of `pandas.read_csv()`

---

### Table Operations

#### Create New Table

```pivotal
# Copy from existing table
table filtered_data from sales:
    filter price > 100

# Switch context to existing table
table sales:
    # operations go here...
```


---

### Filtering

Filter rows based on conditions:

```pivotal
table active_users:
    filter status == "active"

table premium_sales:
    filter amount > 1000 and category == "premium"

table regional_data:
    filter region in ["North", "South", "East"]
```

**Supported Operators:**
- Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=`
- Membership: `in`, `not in`

- Logical: `and`, `or`

---

### Selecting Columns

Choose specific columns to keep:

```pivotal
table customer_summary:
    select customer_id, name, email

table sales_metrics:
    select product, quantity, revenue, profit_margin
```

---

### Creating/Modifying Columns

Create new columns or modify existing ones using the `set` statement:

```pivotal
# Simple calculation
table sales:
    set total = price * quantity

# Conditional assignment
table products:
    set discount_price = price * 0.9
       where category == "clearance"

# Multiple operations
table analysis:
    set revenue = price * quantity
    set profit = revenue - cost
    set margin = profit / revenue
       where revenue > 0
```

**Expression Syntax:**
- Use pandas `.eval()` syntax
- Reference columns directly by name
- Standard operators: `+`, `-`, `*`, `/`, `**`
- Functions: Any pandas-compatible function

---

### Sorting

Sort data by one or more columns:

```pivotal
# Single column, ascending (default)
table sorted_sales:
    sort amount

# Single column, descending
table top_performers:
    sort revenue desc

# Multiple columns
table ranked_products:
    sort category asc, sales desc, price asc
```

**Sort Orders:**
- `asc` - Ascending (default)
- `desc` - Descending

---

### Merging Tables

Join two tables together:

```pivotal
# Inner join (default)
table combined:
    merge other_table on customer_id

# Left join
table sales_with_customers from sales:
    left merge customers on customer_id

# Join with explicit merge type
table full_data:
    outer merge secondary on id

# Join on multiple keys
table matched:
    merge other on key1, key2
```

**Merge Types:**
- `merge` or `inner merge` - Inner join (intersection)
- `left merge` - Left join (all left, matching right)
- `right merge` - Right join (all right, matching left)
- `outer merge` - Outer join (union)

**Advanced Parameters:**
```pivotal
table complex_merge from table1:
    left merge table2
       left_on id
       right_on customer_id
       suffixes ["_left", "_right"]
```

**Parameters:**
- Accepts all keyword arguments of `pandas.merge()`

---

### Pivot Tables

Create pivot tables with aggregations:

```pivotal
# Basic pivot
table sales_pivot:
    pivot on amount
       rows product
       cols region
       agg sum

# Multiple value columns
table multi_metric_pivot:
    pivot on revenue, quantity
       rows category
       cols quarter
       agg sum, mean

# Complex pivot
table detailed_summary:
    pivot on sales, profit, units
       rows product, category
       cols region, quarter
       agg sum, mean, count
```

**Aggregation Functions:**
- `sum` - Sum of values
- `mean` / `avg` - Average value
- `count` - Count of values
- `min` - Minimum value
- `max` - Maximum value
- `median` - Median value
- `std` - Standard deviation
- `var` - Variance
- `first` - First value
- `last` - Last value

---

## VS Code Extension

### Features
- **Syntax Highlighting** - Color-coded keywords, operators, and functions
- **Intelligent Autocomplete** - Context-aware suggestions for tables, columns, and operations
- **Code Execution** - Run Pivotal code directly from the editor
- **Snippets** - Quick templates for common patterns

### Installation

Copy the `pivotal_vscode` folder to your VS Code extensions directory:
```
%USERPROFILE%\.vscode\extensions\  (Windows)
~/.vscode/extensions/               (Mac/Linux)
```

### Keyboard Shortcuts

- **`Shift+Enter`** - Execute entire file in Python Interactive
- **`Ctrl+Enter`** - Execute selected text in Python Interactive

### Commands

- **Pivotal: Execute File** - Run the current `.pivotal` file
- **Pivotal: Execute Selection** - Run selected Pivotal code
- **Pivotal: Reinitialize Parser** - Reload the DSL parsernd

---

## API Reference

### DSLParser Class

```python
parser = pivotal.DSLParser(backend="pandas")
```

#### Methods

##### `parse(code: str) -> list`
Parse DSL code and return the Abstract Syntax Tree (AST).

```python
ast = parser.parse(dsl_code)
```

##### `generate_code(ast: list) -> list`
Convert AST to executable Python code.

```python
python_code = parser.generate_code(ast)
```

##### `execute(code: str, globals_dict: dict, verbose: bool = True) -> dict`
Parse and execute DSL code in one step.

```python
tables = parser.execute(dsl_code, globals(), verbose=True)
```

**Parameters:**
- `code` - Pivotal DSL code string
- `globals_dict` - Namespace to execute in (use `globals()`)
- `verbose` - Print execution details (default: True)

**Returns:** Dictionary of table names to DataFrames

##### `export(code: str) -> str`
Export DSL code as standalone Python script.

```python
python_script = parser.export(dsl_code)
print(python_script)
```

---

## Examples

### Example 1: Sales Analysis

```pivotal
# Load sales data
load sales sales_data.csv
   header 0

load products product_catalog.csv

# Filter high-value sales
table high_value from sales:
    filter amount > 500
    select customer_id, product_id, amount, date

# Merge with product info
table enriched_sales from high_value:
    left merge products on product_id
    select customer_id, product_name, category, amount

# Calculate metrics
table analysis from enriched_sales:
    set revenue = amount
    set is_premium = amount > 1000

# Create summary pivot
table category_summary from analysis:
    pivot on revenue
       rows category
       cols is_premium
       agg sum, mean, count
```

### Example 2: Customer Segmentation

```pivotal
# Load customer data
load customers customer_data.csv
load transactions transaction_log.csv

# Calculate customer metrics
table customer_stats from transactions:
    set total_spent = amount
    
# Aggregate by customer
table customer_summary:
    # Group operations would go here
    
# Segment customers
table segments from customer_summary:
    set segment = "low"
    
table high_value from segments:
    set segment = "high"
       where total_spent > 1000
       
table medium_value from segments:
    set segment = "medium"
       where total_spent > 500 and total_spent <= 1000
```

### Example 3: Time Series Analysis

```pivotal
# Load time series data
load timeseries sensor_data.csv
   header 0

# Filter by date range
table recent_data from timeseries:
    filter date >= "2024-01-01"
    select sensor_id, date, temperature, humidity

# Calculate rolling metrics
table with_metrics from recent_data:
    set temp_fahrenheit = temperature * 9/5 + 32
    set comfort_index = temperature * 0.7 + humidity * 0.3

# Sort chronologically
table chronological from with_metrics:
    sort date asc, sensor_id asc

# Create pivot by sensor
table sensor_pivot from chronological:
    pivot on temperature, humidity
       rows date
       cols sensor_id
       agg mean, min, max
```

### Example 4: Data Quality Check

```pivotal
# Load data
load raw_data input.csv

# Check for issues
table quality_check from raw_data:
    filter status != "invalid"
    select id, name, value, category

# Remove duplicates and standardize
table cleaned from quality_check:
    set name_clean = name
    set value_normalized = value / 100

# Filter outliers
table final_data from cleaned:
    filter value_normalized > 0 and value_normalized < 10
    sort id asc
```

---

## Comments

Pivotal supports single-line and multi-line comments:

```pivotal
# This is a single-line comment

-- This is also a single-line comment

/*
This is a
multi-line comment
*/

load data file.csv
   # Comments can appear anywhere
   header 0  -- Even after code
```

---

## Tips & Best Practices

### 1. **Use Descriptive Table Names**
```pivotal
# Good
table high_value_customers:
    filter total_spent > 1000

# Less clear
table t1:
    filter total_spent > 1000
```

### 2. **Chain Operations Logically**
```pivotal
table analysis from raw_data:
    filter status == "active"    # First filter
    select id, name, value        # Then select needed columns
    set normalized = value / 100  # Then calculate
    sort normalized desc          # Finally sort
```

### 3. **Use Indentation Consistently**
Pivotal uses indentation to define operation blocks. Use spaces (not tabs) for consistency.

### 4. **Break Complex Operations into Steps**
```pivotal
# Instead of one giant table with many operations
table step1:
    filter condition1
    
table step2 from step1:
    merge other_data on key
    
table final from step2:
    select needed_columns
```

### 5. **Test Incrementally**
Execute code step-by-step in an interactive session to verify each operation.

---

## Troubleshooting

### Common Issues

**Import Error: `No module named 'pivotal'`**
- Ensure you have installed the package: `pip install .`
- Check that your virtual environment is active

**Parse Error: Unexpected indentation**
- Check that indentation is consistent (use spaces, not tabs)
- Ensure nested operations are properly indented

**Table Not Found Error**
- Make sure to `load` or create a table before referencing it
- Check table name spelling

**Column Not Found Error**
- Verify column exists in the table
- Check for typos in column names
- Use `print(table_name.columns)` to see available columns

---

## Advanced Usage

### Export to Python Script

Convert Pivotal code to a standalone Python script:

```python
dsl_code = """
load data input.csv
table analysis:
    filter value > 100
"""

python_script = parser.export(dsl_code)

# Save to file
with open('generated_script.py', 'w') as f:
    f.write(python_script)
```

### Programmatic Execution

```python
# Execute and capture results
parser = pivotal.DSLParser()
tables = parser.execute(dsl_code, globals(), verbose=False)

# Access specific tables
if 'analysis' in tables:
    df = tables['analysis']
    print(df.describe())
```

### Custom Backends

Currently, Pivotal supports pandas. Future versions may support other backends like Polars or DuckDB.

---

## Contributing

Contributions are welcome! Areas for improvement:
- Additional aggregation functions
- More merge options
- Group-by operations
- Window functions
- Integration with more data sources

---

## License

[Specify your license here]

---

## Authors

[Your name/team here]

---

## Version History

- **v0.1.0** - Initial release
  - Basic load, table, filter, select operations
  - Merge and pivot support
  - VS Code extension with autocomplete

---

## Contact & Support

For questions, issues, or feature requests, please [open an issue] or contact [your contact info].
