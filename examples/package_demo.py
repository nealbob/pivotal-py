# Package management demo
# Run this file to see start / save / load all in action.
# Set PIVOTAL_PATH to the root of your pivotal-py checkout, e.g.:
import pivotal 
import os

_ppath = os.environ.get('PIVOTAL_PATH', '.')
data_path = _ppath + '/examples/data/data.csv'
output_path = _ppath + '/output'

#%%
%%pivotal
start "sales_demo"
    path :output_path
    title "Sales Demo Package"
    format csv
    style reports
        figsize 15 8
        grid True
#%%
"""
-- Load raw data from disk
load sales :data_path

#%%

-- Clean and enrich
df clean from sales
    filter price > 0
    assign revenue = price * quantity
    sort revenue desc

-- Summarise by category
df summary from clean
    group by category
        agg sum revenue as total_revenue, mean revenue as avg_revenue, count revenue as n

df summary
    plot bar
        x category
        y total_revenue
        title "Total Revenue by Category" 
 
-- Persist both tables to the package
save
"""