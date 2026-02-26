import pandas as pd

invoices = pd.read_csv('invoices.csv')
customers = pd.read_csv('customers.csv')

invoices = invoices.query('invoice_date >= "1970-01-16"')
invoices['transaction_fees'] = invoices.eval('0.8')
invoices['income'] = invoices.eval('total - transaction_fees')
invoices = invoices.query('income > 1')

summary = invoices.copy()
summary = summary.groupby(['customer_id']).agg(total_mean=('total', 'mean'), sum_income=('income', 'sum'), ct=('total', 'count')).reset_index()
summary = summary.sort_values(['sum_income'], ascending=[False])
summary = summary.merge(customers, on='customer_id', how='left')
summary["name"] = summary["last_name"] + ", " + summary["first_name"]
summary = summary.loc[:, ['customer_id', 'name', 'sum_income']]