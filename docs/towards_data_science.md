# The Problem with pandas Is Not Performance. It’s Cognitive Overhead.

*Faster dataframe engines are nice, but they don't reduce the amount of syntax an analyst has to hold in their head.*

For years, the conversation about pandas has focused on performance.

[As its creator has acknowledged](https://wesmckinney.com/blog/apache-arrow-pandas-internals/), pandas' foundations were not built for today’s data workloads. Over time, pandas has made real progress in this area. The recent [pandas 3.0](https://pandas.pydata.org/community/blog/pandas-3.0.html) was a significant step forward, with PyArrow-backed string types offering big performance gains. Meanwhile, Polars and DuckDB have shown what can be achieved when modern columnar data structures are part of the design from the beginning.

But performance is only one cost in data analysis. For many everyday tasks, back-end performance is a secondary concern. The dataset fits in memory. The calculation finishes in a second. But the analyst has to take time to remember an API, rearrange brackets, look up an aggregation pattern and check whether a grouping key has quietly become an index.

The CPU is idle. The human is not.

The deeper problem with pandas—and, to different degrees, with most dataframe APIs—is [cognitive overhead](https://github.com/zakirullin/cognitive-load).

## The tax hidden in ordinary code

Consider a simple task: keep positive sales, calculate a margin, summarise by region and sort the result:

```python
summary = (
    sales.loc[sales["revenue"] > 0]
    .assign(margin=lambda df: df["revenue"] - df["cost"])
    .groupby("region", as_index=False)
    .agg(
        total_revenue=("revenue", "sum"),
        average_margin=("margin", "mean"),
    )
    .sort_values("total_revenue", ascending=False)
)
```

This is not bad pandas. It is not a deliberately awful example assembled to win a syntax comparison. An experienced pandas user can read it without difficulty.

But notice how much of the expression is about negotiating with the API rather than detailing actual logic:

- A column is sometimes `sales["revenue"]`, sometimes `df["revenue"]`, and sometimes the string `"revenue"`.
- Creating a column requires `assign` and a lambda if we want to preserve the method chain.
- A named aggregation is expressed as a tuple whose order is column first, function second.
- Descending order is expressed by setting an `ascending` option to `False`.
- The behaviour of the grouping key depends on `as_index`, a parameter whose significance is not obvious from the analytical task.

None of these details is individually difficult. But each one consumes a small piece of human working memory that could otherwise be used to think about data.

pandas indexes are a good example of this tension. The familiar appearance of `.reset_index()` after a group-by is not just a few extra keystrokes; it is an annoying distraction. But of course, backward compatibility limits how radically a mature library can redesign its surface.

## “AI can write it now” is only half an answer

Who cares if pandas syntax is less than perfect, you might say. AI agents can generate the pandas code for now, so what does it matter? Yes, large language models can save a great deal of time. But generating code is only one part of analytical work.

Coding for data analysis is different from software development. It often begins with a question that changes as soon as the first result appears. You filter the data, notice something unexpected unexpected, inspect it, revise the grouping, discover missing values, make a chart and then realise that your original question was the wrong one.

The workflow is not:

> specification → code → finished product

It is closer to:

> question → transformation → result → new question → new transformation

That loop is exploratory, creative and interactive. In this setting, there's value in a human being able to transform data with minimal latency. I expect that many analysts are still finding themselves typing small pieces of pandas code into a notebook, even if they now trust AI to write larger functions or modules.

## Readability matters

Secondly, while AI reduces the cost of typing, it doesn't remove the cost of reading, checking and understanding.

A Python data pipeline is also documentation. It tells a colleague—or your future self—what was filtered, which variables were created, and where the final number came from. The easier that path is to follow, the easier it is to review assumptions and catch mistakes.

Boilerplate weakens that documentation by lowering the signal-to-noise ratio. The business logic is still present, but it is surrounded by dataframe names, column selectors, quotation marks, lambdas, aliases and API-specific options.

Let's be honest: reading other people's pandas code, especially code that was built through an interactive session can be rather painful. There is no doubt that pandas code is not the most concise or readable way to express the underlying logic of a data pipeline.

The rise of AI-generated code only strengthens this argument. If more code is going to be produced automatically, humans need representations that make the generated logic easy to inspect.

## The enduring popularity of visual data tools

If you're still not convinced that any of this matters, think for a minute about the popularity of visual data tools.

Excel remains embedded in analytical work across almost every industry. Tableau, Power BI, KNIME, Alteryx, Metabase, Orange, RapidMiner and many other products offer different variations on the same promise: touch the data more directly, see feedback quickly and avoid having to translate every thought into a general-purpose programming API.

I use Excel a lot. There are plenty of occasions when dropping a small dataset into a pivot table is faster than writing a pandas pipeline. Visual tools can be more intuitive and help to reduce the latency from thought to result.

Of course, code has a purpose. A script provides an audit trail from raw data to result. It can be reviewed, tested, versioned, rerun and shared. This is why teams move critical work out of spreadsheets in the first place. A lot of modern visual tools now also generate pandas code for this very reason, including [Data Wrangler](https://code.visualstudio.com/docs/datascience/data-wrangler) and [Mito](https://docs.trymito.io/how-to/using-the-generated-code).

But if "code-as-documentation" is kind of the point, why are we using boilerplate-laden pandas code as our language of choice? Why not something that is easier for humans to read?

## Why does every dataframe need a new dialect?

The cognitive cost is not confined to pandas.

Move to Polars and the underlying analytical ideas remain the same: select, filter, derive, group, aggregate, join and sort. But the syntax changes. Move from Python to R, Julia or MATLAB and it changes again.

There might be good reasons for these differences, but from an analyst’s perspective, it is strange that the same logical operation must be relearned as a new dialect in each enviornment. As a data analyst, statistical work might take me to R. Simulation or optimisation may take me to Julia or MATLAB. A larger-than-memory transformation may take me to DuckDB. Why should this also require changing the grammar in which I express `group by region, then sum revenue`?

## DSLs to the rescue?

Most dataframe awkwardness comes from embedding a data-transformation grammar inside a general-purpose programming language.

Python needs to distinguish variables, attributes, strings, list indexing, function calls and assignment. A dataframe library has to construct its language from those same pieces. That is why column names become strings, why the active dataframe is repeatedly referenced and why a simple expression can accumulate punctuation:

```python
sales.loc[sales["revenue"] > 0, "margin"] = (
    sales["revenue"] - sales["cost"]
)
```

A domain-specific language (DSL) can make different choices because it has a narrower job. SQL is the strongest evidence this approach works. It gave data work a shared vocabulary that survived changes in hardware, vendors and host programming languages. Its durability is not an accident.

But SQL may not be the final answer for every analytical workflow. SQL was designed for querying relational databases, not the full interactive loop of loading local data, transforming it, plotting it, calling a model and saving a collection of outputs. Its traditional clause order does not read as a top-to-bottom pipeline, and complex work often expands into nested queries or chains of  expressions.

Projects such as [PRQL](https://prql-lang.org/) and [pipe syntax in SQL](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/pipe-syntax) improve SQL’s top-to-bottom readability considerably, while DuckDB has made SQL feel at home in local Python workflows.

DuckDB's popularity is itself an interesting case study. Its [2024 user survey](https://duckdb.org/2024/10/04/duckdb-user-survey-analysis) showed that many users do not actualy have larger-than-memory datasets, and beyond performance they value DuckDB for its ease of use and SQL support.

But DuckDB, PRQL and piped-SQL are utimately still extensions of regular SQL. DuckDB can transform tables fine, but plotting and other common analytical operations still have to be done in Python, which means switching back to a dataframe API. DuckDB may execute the transformation fast enough, but from an analyst's perspective, switching back and forth is going to interrupt flow.

## Pivotal as an experiment

These concerns are what led me to build [Pivotal](https://www.pivotal-lang.org/), an open-source DSL for data analysis in Python.

Pivotal is not a replacement for the Python ecosystem. It is a compact way to express common analytical operations, which can then compile to Python code using pandas, Polars or DuckDB. The aim is to separate the logic the analyst writes from the engine that executes it.

The earlier pandas example becomes:

```pivotal
with sales as summary
    filter revenue > 0
    margin = revenue - cost
    group by region
        agg sum revenue as total_revenue, mean margin as average_margin
    sort total_revenue desc
```

The difference is not just character count. Each line corresponds to an analytical idea. The active table is declared once. Columns are columns rather than strings. The grouping and its aggregations form a visible block. Reading from top to bottom follows the order in which the analyst is thinking.

Because Pivotal compiles to ordinary Python, its results remain available to the rest of the Python ecosystem. Pivotal also includes commands for plots, tables and saving outputs, while regular Python remains available as an escape hatch when the DSL is not the right tool.

Of course, Pivotal might not be the one syntax to rule them all. It is a young project with its own trade-offs. Users have to learn it. Tooling is limited (VS Code and Jupyter Lab only) and community support is non-existent. pandas has a lot of inertia and for teams with established codebases (including my own) a new language is a tough sell.

But Pivotal is an experiment in a design space that has been underexplored. We benchmark execution time carefully. We should also care about comprehension time, error visibility and the number of concepts an analyst must keep in their head to express a simple transformation.

## Beyond programming languages

It's an interesting time to be alive. When I started my career in the early 2000s, we used Excel and Visual Basic for data wrangling, and paid software like SAS and Stata for analysis. Python was still niche (at least in my circles), and [pandas](https://pandas.pydata.org/about/index.html) and [scikit-learn](https://scikit-learn.org/stable/about.html) were years away. Since then, the Python data ecosystem has become entrenched across academia, industry and government IT environments.

While a lot has changed, this seems tame in comparison with the disruption AI is poised to unleash.

The rise of AI coding agents over the past year has a lot of people wondering about the future of programming. Will we see new languages and DSLs evolve for interacting with agents? Or should agents produce more abstract or visual representations of logic instead of code, like a data pipeline diagram (e.g., an Airflow or KNIME style [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)).

What about the world of visual data tools? At a time when computers understand human speech, point-and-click interfaces are starting to feel a bit quaint. At the same time, we might see a convergence of these products, with tools that mix natural language, visual interaction and code generation seamlessly.

I don't know what the winning interface will look like, but I know that we can do better. Because `df.groupby("year").reset_index()` cannot be the peak of human achievement.


---

*Disclosure: I am the creator of Pivotal, which is an open-source project.*
