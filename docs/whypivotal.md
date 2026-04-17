# Pivotal: A simpler way to work with data in Python

or

# Introducing Pivotal: a data analsyis language for Python

Python has become a standard platform for data anlaysis, particuarly in corporate and government teams. The draw of Python understandable: it's a great language, well known by all, with an excellent data ecosystem, and its free.  

I love Python. But I  don't love Pandas. By its creators admission Pandas syntax has some strange quirks and is overly verbose. While anlaysts flock to Python for the powerfull analysis tools, they invariably spend much of there time doing basic data wrangling, which in Pandas is harder than it should be.

Pivotal is my attmept to address this. Pivotal is a Domain Specfic Language (DSL) for data anlaysis, with a consise syntax that compiles into Python code (using either Pandas, Polars or DuckDB dataframe backends). Pivotal is designed to support interactive Python workflows with a language for data transformation that is faster to type and easier read, while still operating over Python data strucutres and integrating tightly with Python code.

## Two examples

Lets take a look...

    ```pivotal
    load "daily_climate.csv" as climate
    load "crop_data.csv" as crops
    
    python grow_min = 8; grow_max = 32; crop_season = [4, 10]

    with climate as climate_features
        year = year(date)
        month = month(date)
        filter month between :crop_season
        grow_degrees = 0
        grow_degrees = (max_temp + min_temp) / 2 - :grow_min
            where max_temp < :grow_max and min_temp >= :grow_min
        group by year, region
            agg sum grow_degrees as gdd, sum rain as grow_rain

    with crops as training_data
        filter crop_type == "wheat" and area > 0
        yield = production / area
        inner merge climate_features on year, region
        select year, area, yield, gdd, grow_rain, region

    python
        from sklearn.linear_model import LinearRegression
        X = training_data[["year", "area", "gdd", "grow_rain"]]
        training_data["yield_hat"] = LinearRegression().fit(X, training_data["yield"]).predict(X)
    end

    with training_data
        pivot plot line nat_pred_vs_actual
            x year "Year"
            y wmean yield area, wmean yield_hat area "Wheat yield (t/ha)"
    ```

While styliysed this example depicts a common data science workflow: loading data and feature engineering, then model training, followed by processing of results into plots, tables and other outputs.  As in this example, real workflows often involve a high propotion of code (and human effort) being spent on data processing relative to acutal modeling.

Pivotal has a declarative syntax similar to SQL while incopratating aspects Python (pandas) and R (dplyr) grammer. Becuase Pivotal compiles to Python it is easy to access Python objects and functions within Pivotal code, and to intesperse Python and Pivotal either in script files (as above) or Notebooks.

Pivotal has been deisgned to be easy-to-type, with minimal use of punctaution, symbols or brackets.  This speed is important in the context of typical interactive exploratory data work. The below example compares Pivotal to comparable 'human-written' Python code (in the context of a Jupyter Notbook). In this example, Pandas code equires around 50% more charecters and 70% more key strokes.

Importanly Pivotal's syntax is also much more human readable, which is important for colaboaration among teams and potentialy for interaction with AI agents (but more on that another time).

=== "Pivotal"

    ```python
    import pivotal
    ```

    ```pivotal
    %%pivotal
    load "invoices.csv" as invoices
    load "customers.csv" as customers

    with invoices
        filter invoice_date >= "1970-01-16"
        transaction_fees = 0.8
        income = total - transaction_fees
        filter income > 1

    with invoices as summary
        group by customer_id
            agg mean total, sum income as sum_income, count total as ct
        sort sum_income desc
        left merge customers on customer_id
        name = last_name + ", " + first_name
        select customer_id, name, sum_income

    save "my_analysis"
        path "~/projects/output"
    ```

=== "Pandas"

    ```python
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
    summary.to_csv("~/projects/output/my_analysis.csv", index=False)
    ```

    Readable with method chaining, but requires knowing the `agg` dict-of-tuples syntax, that `.reset_index()` is needed after `groupby`, and that column assignment must happen outside the chain.

=== "Polars"

    ```python
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
    summary.write_csv("~/projects/output/my_analysis.csv")
    ```

    Fast and expressive, but every column reference requires `pl.col()` and every literal `pl.lit()`. The ceremony adds up across a longer pipeline.

=== "DuckDB / SQL"

    ```python
    # Setup cell (once per notebook)
    %load_ext sql
    %sql duckdb://
    ```

    ```sql
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
    ```

    ```sql
    %sql copy summary to '~/projects/output/my_analysis.csv' (header)
    ```

    [JupySQL](https://jupysql.ploomber.io/) provides `%%sql` cell magic backed by DuckDB, which means you can write clean SQL directly in a notebook cell.

    CTEs make this readable and the `%%sql` magic keeps the notebook experience clean. The gaps are multi-step mutations (each requires a new CTE), no built-in file export, and results need a Python cell to do anything further with them.

=== "PRQL"

    ```python
    # Setup cell (once per notebook)
    %load_ext pyprql.magic
    %load_ext sql
    %sql duckdb:///:memory:
    %sql create view invoices as select * from read_csv_auto('invoices.csv')
    %sql create view customers as select * from read_csv_auto('customers.csv')
    ```

    ```
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
    ```

    ```python
    summary.to_csv("~/projects/output/my_analysis.csv", index=False)
    ```

    [PRQL](https://prql-lang.org/) (Pipelined Relational Query Language) compiles to SQL. Its pipeline style is the closest conceptually to Pivotal. The [pyprql](https://github.com/prql/PyPrql) package provides a `%%prql` Jupyter magic backed by DuckDB, equivalent to `%%sql`.
    
    PRQL reads very naturally as a pipeline — arguably the most readable of the SQL-family options. The `%%prql` magic removes the need for Python glue around the query. File export still requires a separate Python cell.

---

| | Pivotal | Pandas | Polars | DuckDB/SQL | PRQL |
|---|---|---|---|---|---|
| Lines | 18 | 23 | 29 | 32 | 27 |
| Characters | 547 | 866 | 911 | 769 | 685 |
| Key presses | 542 | 937 | 983 | 753 | 738 |
| Tokens | 103 | 256 | 299 | 176 | 169 |


## The Pandas problem and the DSL solution

Pandas limitations have been well-documented, most notably by its creator [Wes McKinney](https://wesmckinney.com/blog/apache-arrow-pandas-internals/). Firstly, there are the quirks: including Pandas use of indexes which most peoplpe find Unnecessary, resulting in biolerplate like `.reset_index()` or `as_index=False` just to work around them. Then there are the performance limiations owing to Pandas dataframes use of Numpy arrays.

There are of course alternatives. Polars offers better perofrmance (though the syntax is even more verbose). The Python Ibis API has better syntax, and if we are being honest, the R [tidyverse](https://tidyverse.org/) offers a better expereince for interactive data work than anything in Python.

However, all of these options suffer from the same constraint of embeding a data processing API within a gerneal purpose language.  This leads to  redundant boilerplate like wrapping column names in quotations and explictly referencing dataframes each time, for example

```python
mydata.loc[mydata.columnA > 0, "columnB"] = mydata["columnB"] / mydata["columnA"]
```
 
compared with: 

```pivotal
    with mydata
        columnB = columnB / columnA
            where column A > 0
```

The longevity of SQL tells us something about the value of DSLs for data work. In recent years, SQL has become better integrated with Python through libraries like DuckDB, while new piped-SQL syntax including [PRQL](https://prql-lang.org/) offer a more linear Python/R style of working.

In many respects, DuckDB and PRQL address some of the same problems as Pivotal, just from the opposite direction: trying to modfy SQL to bring it closer to Python, rather than building a native Python workflow that is more SQL like.

But there are limits to how far you can bend SQL to suit Python. DuckDB and PRQL still use a SQL engine built for a very different purpose: queruing large databases. As such SQL in Python is far from seemless, with a lot of swtiching between data structures and mental models.

## How Pivotal works

Under the hood, Pivotal is just a code generator which takes strings of Pivotal syntax and outputs strings of Python code. Pivotal is written in Python, and uses the lark package to parse Pivotal code into an Abstract Syntax Tree AST (i.e., Python dictionary).  From this AST, Pivotal can then generate code for multiple 'backends' including Python Pandas, Polars or DuckDB code (or even SQL CTEs):

```Python
import pivotal
pivotal.export("with mydf\n select columnA, columnB", backend="pandas")

<show output and compate with other backends>
```

The coverage of Pivotal is reasnobly extensive including all basic Pandas / SQL data operations: e.g.,  `filter`, `select`, `sort`, `group by` / `pivot`, `merge` and column-wise expresions along with a range of more complex tasks like window functions and date and string maniuplation (for a full descirption, see the [docs](https://nealhughes.net/pivotal-py/)). Pivotal also includes commands for producing outputs, including plots (via `matplotlib`), tables (via `GreatTables`) and saving to data packages. For any complex tasks that can't be done in Pivotal there is of course an easy Python "excape hatch".

Pivotal has benn built with Jupyter Notebooks in front of mind. The JupyterLab extenion includes ```%%pivotal``` cell magic with syntax highlighting and context aware auto-complete for Pivotal code cells (column and table name completions linked to the active Python session), along with GUI features including interactive object viewer and explorer panes (with AG GRID spreadsheets and table and plot previews). There is also a VS Code extension offering much the same functionality.

JUPYTER LAB GIF

## We need your help

It would be  misleading to say I have built Pivotal myself, given most of the code has been produced by AI (particuarly Claude) which at this point feels more like a collaborator than a servant. This project is well suited to AI given it is so closely realted to exsting langauages LLMs have been trained on, and it is easy to define objective tests. There is a lot more to say about the role of AI here, both in the development and use of DSLs, but I'll leave that for antoher time. While AI greatly simplfies development, there is still a strong need for human guidance, given the whole purpose of a project like Pivotal is to develop a language better suited to human tastes and ways of thinking. The key thing Pivotal needs at this point is more feedback from more humans. So please give Pivotal a go and let me (and Claude) know what you think.






I want to add a section to the docs the outlines the motivation behind Pivotal. The goal is to develop interest among potential users, and to counter obvious concerns, doubts and critasicims that are likely to be raised. I imagine this being a section of the docs, but potentialiy a version of it might later be published as an online article / blog post to advertise Pivotal.

I have drafted below an extensive set of notes. Before attempting to write a complete section I want you to look over my notes and give me some thoughts. I want you to provide honest feedback including:

    - Is there anything I am missing, any strengths of Pivotal I should say more about? OR any potential critascisims of Pivotal I am not covering off on
    - In cases where I have explcitly noted that I could do with some addtional information / reseach to support an argument can you help to provide data / examples or at least give me some advice on where to look.
    - Do you think these arguments are likely to be persuasive with Pivotal's potential audience, can you identify any weaknesses in my case
    - Do you think the strucutre and framing of the article is sound, the sequence of the arguments, do you have any suggestions on the order or priority of the arguments, is there anything i should cut
    - Do you have any thoughts on how i might translate this into a blog post or multiple blog posts (I was thinking either of 1 which summarises all of it, or 2, one on pivotal and one more on the AI angle)
  
Currently imagine the article being structured like this...


    - Breif Introduction section 

        Nutshell argument. Python is becoming the standard enviornment for data anlysis. Basic data operations are the core of this work flow, and much of this is done with Pandas. Pandas syntax has some limitations, Pivotal is a DSL designed to support a python first workflow with a simpler syntax that is easeier to type (important for interactive work) and easier to read (imporant for validation / collaboration). But you  say: why not just use SQL? or Why write code at all when you can use Gui tools or better yet AI?(here is my attemtp to address all that)

    - Section 0: What is Pivotal. This section would give a breif introduction to pivotal syntax. Explain how it works - parsing into multiple backends. I think we could show the comparison example from the docs we could also show the at a glance example. Here we would explain the syntax design philosphy - concise - fast to type - avoding special characters and anything that slows down flow, but also being readable. Inspired to some extent by SQL, ultimatly it is a blend of Python and SQL style syntax, designed to be a "Pythonic" data analysis language . It has VS code and Jupyer lab extensions. Well suited to notebooks where you can use cell magic. Very easy to integrated with Python (python variables and functions embeded etc.) because it compiles to pure Python

    - Argument 1: Python is quickly becoming (if it isnt allready) the primary place where data anlaysis happens, Python is the dmoninant programming language in the world, it has large data eco-system espeacialy for data science / machine learning (but also for statistics, modeling / simulation etc.), IT seems to be diplsacing R to some extent (might need some data on this). Combined R and Python are displaceing older closed source software (SAS, stata, Eviews, excel). Many GUI data tools are supporting Python now (even excel). Argument 1b: A lot of applied data anlysis work involves simple data manipualtion. If you look at most projects the core analysis (the fun part) is often a small % of the ctotal ode base, a lot of the work is data clearning and result processing, commonly in Pandas. Wonder if we can get some data on this, but I imagine that there is a lot of Pandas code being written in the world, (can we check this in some objective way like via GitHub or something).
- 
    - Argument 2: Pandas has some limitations. Pandas syntax is not ideal, but because it is so ubiqutous now it is hard to change. Things like indexes (needing to do reset_index() post groupby etc) are widely considered flaws (need some backup links / evidence on this). Pandas also has performance issues which matter as datasets are becoming bigger. Polars is faster but has even more vebose syntax. A lot of people would agree that R tidyverse does a slightly better job, more intuituve and more consise syntax, better suited to interactive work than Pandas (need some backup evidence for this). But R is also not ideal either as is like Python a full featured language (not a DSL). There are benefits of a DSL for data transformation (as demonstrated by ongoing popularity of SQL) where you can set the context ("I am working on a given datafame right now") such that the syntax from that point on is more concise (i.e., colC = colA*colB not mydataset["colC"] = mydataset["colB"]*mydataset["colA"]).  Argument 2b: consiseness matters because data anlysis is done interactively: it is an exploratoty process where you want to "play" with the data, speed  is useful in this case (as demonstrateed by there being so many GUI based tools for data work).  Consiseness also matters for readablity. Pandas code is not always nice to read (Pandas code involves some boilerplate, it is not really pure logic / 'code' but more like a transcript of all the data operations you made in your session) not always easy to follow some one eless pandas code.
    
    - Argument 3: Why not use SQL

    SQL is a DSL for data that is allready widley used why not just use that. Well, SQL was built for a specific purpose: querying databases, not data anlaysis workflows / pipelines. SQL is built for maniuplating large databases rather than in-memory data, with a focus on performance more so that fast interactive use. Becuase it is a DSL it has some syntax advantges, but also carrries some baggage from its differnt use case / history. Couinter point is that SQL is evolving. DuckDB allows you to run SQL queries on in-memory data without an extternal database - modern SQL varients like 'Piped SQL' and PRQL take some ideas from R and Python (linear data trnasformation flow one-step at a time) and arive at something that is better suited for interactive data anlaysis, and is easier to follow. In some ways, emergence of Piped SQL/ PRQL and duckDB are appraoching the same problem from a different direction - make SQL more python like (rather than make python more sql like). But there are limits on how farm you can go in this direction. While you can use SQL / duckDB in ppython it requires a bridge - you have to change data strucutres between database format and in-memory dataframes, and SQL still uses the same query engine so ts limited to doing certain things (you cant do plots you cant do fillna etc in SQL - are there any better examples) so you have to swtich back and forth reguarly.

    - Argument 4: Why not use AI

    The ability of AI to code is impressive and it has improved a lot in the last 6 months. But coding for data analysis is not yet dead. All  around the world people are still writing  SQL and Pandas code as we speak. Why? First, there are security issues, a lot of data is in secure IT environments and staff do not have approval to connect data to external AI. This will prpbably change overtime as local AI / on premises becomes more common. Second, data anlaysis is often an interactive / exploraorty exercise. ITs not like software engineering where you can outline a precises spec and ask for it to be built, its often organic and the analysist gets new ideas on the go. That said AI is being used to support this work increasingly, and LLMs are very good at writing pandas code becuase they have been trained on so much of it. Pivotal could be complementary to the emergence of AI. First, humans may be writing less code in the near future but we may still need to read / validate machine generate code - Pivotal could help by being easier to read than SQL or pandas. Second, AI is making it a lot easier to create and maintiain software including new langauges. In the past, Building a new language / DSL that is slightly more convenient would not be worth the effort, but the effort is so much lower now. Pivotal is a project well suited to "vibe coding" because it is so close to things the LLMs have been trained on.
    
    --- SIDE IDEA: There is a sense in which Pivotal has smiilarities with markdown in this context. It is a super simple human readable format, that  is harder for machines to parse (via usual rule based parsers), just like markdown is harder to parse for machines than html or latex. But with AI simple text formats that can be easliy generated and read by LLMs and easily understood by humans start to become more valuable than (what used to be considered) more machine friendly formats (like html etc.). LLM is making simple text formats (markdown, CLI script, regex etc.) more useful, LLMs now mean that conciseness and readability is more important. 

    - Argument 5: Why not use point-and-click GUIs

    - In addtion to DuckDB / piped SQL a key alternative to Pivotal are GUI-based no-code or low code data tools, like Power-BI, BambooLib, Data Wrangler, Knime even Excel. Particuarly tools like bamboolib and datawrangler (and related alternatives) which take point and click actions and generate pandas code. The profilferation of these tools suggests there is clearly demand for simpler ways to genearate pandas code.  There has to be some doubt however if point and click interfaces driven by mouse are the way of the future given the way AI is progressing. LLMs make plain text formats more powerfull, arguably generating pivotal code via typeing or prompting an AI may may be easier and more future proof than pointing and clicking in Menus. And while there has been a profilferation of these GUI based tools a lot of people are still chooseing to write Pandas code either directily or indirectly via GUI / AI. Some extenions to Pivotal might help stengthen this argument. 1. It would probably not be that hard to get AI to produce pivotal code (via a skill doc or syntax parser) 2. If pivotal could work in a round trip way (pandas - > pivotal code)

ideas...

ABARES background and history

Research topics

Publications (report and data)

Data sources (sometimes in publications)

Models 
    history / versions

Partnerships and stakeholders

DAFF
RDCs
CEBRA
ABS ag data
Fish
Forrests
CSIRO