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
  - [Data Cleaning](#data-cleaning)
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

df high_value_sales from sales:
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

Load data files into named tables. The file format is detected automatically from the extension:

```pivotal
# CSV (default)
load sales data.csv

# Excel
load budget report.xlsx

# Parquet
load events events.parquet

# Quoted path
load inventory "data/inventory_2024.csv"
   names ["product", "quantity", "price"]
   header 0
```

**Supported formats:** `.csv`, `.xlsx`, `.xls`, `.parquet`

**Parameters:**
- Accepts all keyword arguments of the relevant pandas reader (`read_csv`, `read_excel`, `read_parquet`)

---

### Table Operations

#### Create New Table

```pivotal
# Copy from existing table
df filtered_data from sales:
    filter price > 100

# Switch context to existing table
df sales:
    # operations go here...
```


---

### Filtering

Filter rows based on conditions:

```pivotal
# Comparison
df active_users from users:
    filter status == "active"

# Logical operators
df premium_sales from sales:
    filter amount > 1000 and category == "premium"

# Membership
df regional_data from sales:
    filter region in ["North", "South", "East"]

# Range (inclusive)
df mid_range from sales:
    filter price between [100, 500]

# String matching
df laptop_sales from sales:
    filter product contains "Laptop"

df recent from logs:
    filter event startswith "login"

df errors from logs:
    filter message not contains "warning"
```

**Supported Operators:**
- Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=`
- Membership: `in`, `not in`
- Range: `between [lo, hi]`
- String: `contains`, `not contains`, `startswith`, `endswith`
- Logical: `and`, `or`

---

### Selecting Columns

Choose specific columns to keep:

```pivotal
df customer_summary from customers:
    select customer_id, name, email

df sales_metrics from sales:
    select product, quantity, revenue, profit_margin
```

---

### Creating/Modifying Columns

Create new columns or modify existing ones using the `set` statement:

```pivotal
# Simple calculation
df sales:
    set total = price * quantity

# Conditional assignment
df products from catalog:
    set discount_price = price * 0.9
       where category == "clearance"

# Multiple operations
df analysis from sales:
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
df sorted_sales from sales:
    sort amount

# Single column, descending
df top_performers from sales:
    sort revenue desc

# Multiple columns
df ranked_products from sales:
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
df combined from sales:
    merge other_table on customer_id

# Left join
df sales_with_customers from sales:
    left merge customers on customer_id

# Join with explicit merge type
df full_data from table1:
    outer merge secondary on id

# Join on multiple keys
df matched from table1:
    merge other on key1, key2
```

**Merge Types:**
- `merge` or `inner merge` - Inner join (intersection)
- `left merge` - Left join (all left, matching right)
- `right merge` - Right join (all right, matching left)
- `outer merge` - Outer join (union)

**Advanced Parameters:**
```pivotal
df complex_merge from table1:
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
df sales_pivot from sales:
    pivot
       agg sum amount
       rows product
       cols region

# Multiple aggregations on multiple columns
df multi_metric_pivot from sales:
    pivot
       agg sum revenue, mean quantity
       rows category
       cols quarter

# Complex pivot with multiple functions per column
df detailed_summary from sales:
    pivot
       agg sum sales, mean profit, sum units
       rows product, category
       cols region, quarter
```

**Aggregation Functions:**
- `sum` - Sum of values
- `mean` / `avg` - Average value
- `count` - Count of values
- `min` - Minimum value
- `max` - Maximum value
- `median` - Median value
- `std` - Standard deviation

---

### Data Cleaning

#### Drop Columns

Remove one or more columns:

```pivotal
df clean from sales:
    drop id, internal_ref
```

#### Rename Columns

Rename columns with `as`:

```pivotal
df renamed from sales:
    rename product as item, quantity as qty, unit_price as price
```

#### Handle Missing Values

Fill or drop rows with null values:

```pivotal
# Fill all nulls with a value
df filled from raw:
    fillna 0

df filled_str from raw:
    fillna "unknown"

# Drop rows that contain any null
df complete from raw:
    dropna

# Drop rows where specific columns are null
df complete from raw:
    dropna price, quantity
```

#### Deduplicate

Remove duplicate rows:

```pivotal
# Remove fully duplicate rows
df unique from sales:
    distinct

# Remove duplicates based on specific columns
df unique from sales:
    distinct product, category
```

#### Concatenate Tables

Stack tables vertically:

```pivotal
# Append one table to another
df combined from jan_sales:
    concat feb_sales

# Append multiple tables at once
df all_sales from q1:
    concat q2, q3, q4
```

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
df high_value from sales:
    filter amount > 500
    select customer_id, product_id, amount, date

# Merge with product info
df enriched_sales from high_value:
    left merge products on product_id
    select customer_id, product_name, category, amount

# Calculate metrics
df analysis from enriched_sales:
    set revenue = amount
    set is_premium = amount > 1000

# Create summary pivot
df category_summary from analysis:
    pivot
       agg sum revenue, mean revenue, count revenue
       rows category
       cols is_premium
```

### Example 2: Customer Segmentation

```pivotal
# Load customer data
load customers customer_data.csv
load transactions transaction_log.csv

# Calculate customer metrics
df customer_stats from transactions:
    set total_spent = amount

# Aggregate by customer
df customer_summary from customer_stats:
    group by customer_id
       agg sum total_spent as total_spent

# Segment customers
df segments from customer_summary:
    set segment = "low"

df high_value from segments:
    set segment = "high"
       where total_spent > 1000

df medium_value from segments:
    set segment = "medium"
       where total_spent > 500 and total_spent <= 1000
```

### Example 3: Time Series Analysis

```pivotal
# Load time series data
load timeseries sensor_data.csv
   header 0

# Filter by date range
df recent_data from timeseries:
    filter date >= "2024-01-01"
    select sensor_id, date, temperature, humidity

# Calculate rolling metrics
df with_metrics from recent_data:
    set temp_fahrenheit = temperature * 9/5 + 32
    set comfort_index = temperature * 0.7 + humidity * 0.3

# Sort chronologically
df chronological from with_metrics:
    sort date asc, sensor_id asc

# Create pivot by sensor
df sensor_pivot from chronological:
    pivot
       agg mean temperature, min temperature, max temperature
       rows date
       cols sensor_id
```

### Example 4: Data Cleaning Pipeline

```pivotal
# Load data
load raw_data input.csv

# Drop columns we don't need
df trimmed from raw_data:
    drop internal_id, last_modified

# Rename for clarity
df renamed from trimmed:
    rename cust_nm as customer_name, val as value, cat as category

# Remove rows with missing critical fields
df no_nulls from renamed:
    dropna customer_name, value

# Fill remaining nulls in non-critical fields
df filled from no_nulls:
    fillna "uncategorised"

# Remove duplicates on key columns
df deduped from filled:
    distinct customer_name, value, category

# Filter to valid range and known categories
df final_data from deduped:
    filter value between [0, 10000]
    filter category not contains "test"
    sort value desc
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
df high_value_customers from customers:
    filter total_spent > 1000

# Less clear
df t1 from customers:
    filter total_spent > 1000
```

### 2. **Chain Operations Logically**
```pivotal
df analysis from raw_data:
    filter status == "active"    # First filter
    select id, name, value        # Then select needed columns
    set normalized = value / 100  # Then calculate
    sort normalized desc          # Finally sort
```

### 3. **Use Indentation Consistently**
Pivotal uses indentation to define operation blocks. Use spaces (not tabs) for consistency.

### 4. **Break Complex Operations into Steps**
```pivotal
# Instead of one giant df with many operations
df step1 from raw_data:
    filter condition1

df step2 from step1:
    merge other_data on key

df final from step2:
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
df analysis from data:
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

---

## Contact & Support

For questions, issues, or feature requests, please [open an issue] or contact [your contact info].
