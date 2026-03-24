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
        sum_income=("income", "sum"),
        ct=("total", "count"),
        mean_total=("total", "mean")
    )
    .reset_index()
    .sort_values("sum_income", ascending=False)
    .merge(customers, on="customer_id", how="left")
)

summary["name"] = summary["last_name"] + ", " + summary["first_name"]
summary = summary[["customer_id", "name", "sum_income"]]

summary.to_csv("~/projects/output/my_analysis.csv", index=False)
summary.to_csv("~/projects/output/invoices_new.csv", index=False)
