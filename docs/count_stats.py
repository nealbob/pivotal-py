import re

# Characters that require Shift on a US keyboard
SHIFT_CHARS = set('!@#$%^&*()_+{}|:"<>?~')

examples = {
    "Pivotal": """\
import pivotal
%%pivotal
load invoices "invoices.csv"
load customers "customers.csv"

df invoices
    filter invoice_date >= "1970-01-16"
    transaction_fees = 0.8
    income = total - transaction_fees
    filter income > 1

df summary from invoices
    group by customer_id
        agg mean total, sum income as sum_income, count total as ct
    sort sum_income desc
    left merge customers on customer_id
    name = last_name + ", " + first_name
    select customer_id, name, sum_income

save "my_analysis"
    path "~/projects/output"\
""",

    "pandas": """\
import pandas as pd

invoices = pd.read_csv("invoices.csv")
customers = pd.read_csv("customers.csv")

invoices = invoices[invoices["invoice_date"] >= "1970-01-16"]
invoices["transaction_fees"] = 0.8
invoices["income"] = invoices["total"] - invoices["transaction_fees"]
invoices = invoices[invoices["income"] > 1]

summary = (
    invoices
    .groupby("customer_id")
    .agg(
        mean_total=("total", "mean"),
        sum_income=("income", "sum"),
        ct=("total", "count")
    )
    .reset_index()
    .sort_values("sum_income", ascending=False)
    .merge(customers, on="customer_id", how="left")
)

summary["name"] = summary["last_name"] + ", " + summary["first_name"]
summary = summary[["customer_id", "name", "sum_income"]]

invoices.to_csv("~/projects/output/invoices.csv", index=False)
summary.to_csv("~/projects/output/my_analysis.csv", index=False)\
""",

    "Polars": """\
import polars as pl

invoices = pl.read_csv("invoices.csv")
customers = pl.read_csv("customers.csv")

invoices = (
    invoices
    .filter(pl.col("invoice_date") >= "1970-01-16")
    .with_columns([
        pl.lit(0.8).alias("transaction_fees"),
        (pl.col("total") - 0.8).alias("income")
    ])
    .filter(pl.col("income") > 1)
)

summary = (
    invoices
    .group_by("customer_id")
    .agg([
        pl.col("total").mean().alias("mean_total"),
        pl.col("income").sum().alias("sum_income"),
        pl.col("total").count().alias("ct")
    ])
    .sort("sum_income", descending=True)
    .join(customers, on="customer_id", how="left")
    .with_columns(
        (pl.col("last_name") + ", " + pl.col("first_name")).alias("name")
    )
    .select(["customer_id", "name", "sum_income"])
)

invoices.write_csv("~/projects/output/invoices.csv")
summary.write_csv("~/projects/output/my_analysis.csv")\
""",

    "%%sql": """\
%load_ext sql
%sql duckdb://
%%sql
create or replace table summary as
with enriched as (
    select *,
        0.8 as transaction_fees,
        total - 0.8 as income
    from read_csv_auto('invoices.csv')
    where invoice_date >= '1970-01-16'
),
filtered as (
    select * from enriched
    where income > 1
),
grouped as (
    select
        customer_id,
        avg(total) as mean_total,
        sum(income) as sum_income,
        count(*) as ct
    from filtered
    group by customer_id
)
select
    g.customer_id,
    c.last_name || ', ' || c.first_name as name,
    g.sum_income
from grouped g
left join read_csv_auto('customers.csv') c on g.customer_id = c.customer_id
order by g.sum_income desc
%sql copy summary to '~/projects/output/my_analysis.csv' (header)\
""",

    "PRQL": """\
%load_ext pyprql.magic
%load_ext sql
%sql duckdb:///:memory:
%sql create view invoices as select * from read_csv_auto('invoices.csv')
%sql create view customers as select * from read_csv_auto('customers.csv')
%%prql summary <<
from invoices
filter invoice_date >= @1970-01-16
derive {
  transaction_fees = 0.8,
  income = total - transaction_fees
}
filter income > 1
group customer_id (
  aggregate {
    average total,
    sum_income = sum income,
    ct = count total,
  }
)
sort {-sum_income}
join c=customers (==customer_id)
derive name = f"{c.last_name}, {c.first_name}"
select {
  c.customer_id, name, sum_income
}
summary.to_csv("~/projects/output/my_analysis.csv", index=False)\
""",
}

def indent_to_tabs(text, indent_size=4):
    """Replace leading spaces with tabs so each indent level = 1 key press."""
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip(' ')
        n_spaces = len(line) - len(stripped)
        n_tabs = n_spaces // indent_size
        remainder = n_spaces % indent_size
        lines.append('\t' * n_tabs + ' ' * remainder + stripped)
    return '\n'.join(lines)

def count_keypresses(text, indent_size=4):
    text = indent_to_tabs(text, indent_size)
    count = 0
    for char in text:
        if char == '\n':
            count += 1  # Enter
        elif char == '\t':
            count += 1  # Tab is a single key press
        elif char in SHIFT_CHARS:
            count += 2  # Shift + key
        elif char.isupper():
            count += 2  # Shift + letter
        else:
            count += 1  # regular char inc. space
    return count

def count_tokens(text):
    # Approximate LLM tokenisation: words and punctuation as separate tokens
    return len(re.findall(r'[a-zA-Z0-9_]+|[^\w\s]|\S', text))

def non_blank_lines(text):
    return sum(1 for l in text.splitlines() if l.strip())

indent_sizes = {"PRQL": 2}

print(f"{'':12} {'Lines':>6} {'Chars':>6} {'Keypresses':>11} {'Tokens':>7}")
print("-" * 46)
for name, code in examples.items():
    lines  = non_blank_lines(code)
    chars  = len(code)
    kp     = count_keypresses(code, indent_size=indent_sizes.get(name, 4))
    tokens = count_tokens(code)
    print(f"{name:12} {lines:>6} {chars:>6} {kp:>11} {tokens:>7}")
