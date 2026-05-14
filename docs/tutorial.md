# Tutorial: 10 minutes to Pivotal

This is a short introduction to Pivotal, for more detail see the User Guide or complete Syntax Reference.

In these examples we use VS Code to edit and execute Pivotal files using the VS Code extension (avaialable on VS Code marketplace). For installation details see Getting started.

# Start a '.pivotal' file and load some data 

In VS Code create a new file annd name it with a `.pivotal` extension (the code from this tutorial is available in `tutorial.pivotal` file in the [pivotal-demo](https://github.com/nealbob/pivotal-demo) repo).

To load a CSV data file:

```pivotal
load "data\titanic.csv" as titanic
```

`load` requires a filepath (with or without quotes, windows or linux style paths, local or relative paths or URLs). Data can be in CSV, Parquet or Excel format. Data can also be loaded from SQL databases via the `from' command (see the User Guide for details).

# Getting to know the Pivotal IDE (VS Code extension)

In VS Code you can execute Pivotal interactively (via local Python Kernel & built-in Jupyter Notebook) with <CTRL+ENTER> (full file) or <SHIFT+ENTER> (selected text only). On first execution an interactive Jupyter session will start, then the Pivotal Object explorer and Viewer pane will become visible:

![VS Code screenshot](assets/tutorial1.png)

The object explorer contains a list of all the objects in the current Pivotal session (dataframes, plots, tables, values) for now it will just contain the `titanic` dataframe. The right hand Viewer pane provides a spreadsheet version of the `titanic` table with ability to scroll, sort, filter interactively (without editing the underlying data).

# Modify a dataframe in place

To modfy the `titanic` dataframe we apply commands in a `with titanic` block:

```pivotal
with titanic
    sort Age
```

While the Pivotal Viewer is one way to inspect output, you can also show results within the notebook if prefered.

```pivotal
with titanic
    show head
```

# Make a plot or two
    

with titanic 
    plot hist age_by_survival    
        y Age
        by Survived

with titanic
    pivot plot bar dfdf
        x Pclass
        y mean Survived

# Create new dataframe from existing, filter and select

with titanic as oldest_passengers
    filter Age > 70
    select Age, Name, Survived

# Error handling

with wrong_table
    select Age

with titanic
    select age

# Aggregation

with titanic as titanic_survival_rates
    group by Pclass, Sex
        agg mean Survived

# Pivot tables and Publication ready tables

with titanic_survival_rates
    pivot 
        rows Pclass
        cols Sex
        agg mean Survived

    cast Pclass as string

    table survival_table
        title "Titanic survival rates by class and sex"
        stub Pclass "Passenger Class"
        label female as "F", male as "M"
        spanner female, male "Sex"
        format number 2

# Lists, Scalars, Dicts

list mylist = Age, Survived

scalar myvar = 20

with titanic as temp
    filter Age > myvar
    select mylist
    show head

python
    print(temp.head())
    print(myvar) 
    pylist = ["Pclass", "Parch"]
    pyvar = 3
end

with titanic as temp
    filter Parch > :pyvar
    select :pylist
    show head

# Data mutation and cleaning 

list features = Age, Age2, family, male, Fare

with titanic as X
    family = Parch + SibSp

    male = 1
        where Sex == "male"
        else 0

    median_age = median(Age)
        by Sex, Pclass       
    fillna Age median_age
    
    Age2 = Age**2

    select features 
    
    assert features not null

# Python integration

python 
    import statsmodels.api as sm
    model = sm.Logit(titanic.Survived, X).fit()
    print(model.summary())
    titanic["predicted"] = model.predict(X)
end

python import pivotal; pivotal.update()

# Data package export

save "titanic_results"

# Bulk load

python 
    from pathlib import Path
    p = r"C:\Code_win\pivotal-demo\tutorial\data\AFL\matches"
    files = [f for f in Path(p).iterdir() if f.is_file()]
    print(files)
end

bulk load :files as afl_games

with afl_games 
    team_2_score = team_2_final_goals*6 + team_2_final_behinds 
    team_1_score = team_1_final_goals*6 + team_1_final_behinds
    team_score = (team_1_score + team_2_score)/2 
   
    # Clean team names
    for col in team_1_team_name, team_2_team_name
        col = replace(col, "Kangaroos", "North Melbourne")
        col = replace(col, "Footscray", "Western Bulldogs")

    pivot plot line long_term_scoring_trend
        x year "VFL/AFL season"
        y mean team_score "Mean score per team"
    
    winner =
        where team_1_score > team_2_score: team_1_team_name
        where team_2_score > team_1_score: team_2_team_name
        else "draw"

function ha_games(input, output, col)

    with input as output
        select col, winner, year, date
        win = 1
            where col == winner
            else 0
        select col as team_name, win, year, date
        filter year >= 1990

ha_games(afl_games, home_games, team_1_team_name)
ha_games(afl_games, away_games, team_2_team_name)

with home_games as all_games
    concat away_games

with all_games as all_games_mean    
    group by team_name
        agg mean win
    sort win
    plot barh win_rate_since_1990
        x team_name "Team"
        y win "Win rate since 1990"
        title "Go Cats!"

load data\AFL\lineups\team_lineups_geelong.csv as cats_lineup

with cats_lineup
    select date, team_name, players
    show head

with all_games as cats_with_ablett
    inner merge cats_lineup on team_name, date
    
    with_ablett = "With Ablett"
        where players contains "Gary Ablett"
        else "No Ablett"
    
    era =
        where year<=1996: "Gary Snr - 1990-1996"
        where year>=2002 and year <2011: "Gary Jnr. - 2002-2010 + 2018-2020"
        where year>2018 and year <2021: "Gary Jnr. - 2002-2010 + 2018-2020"

    pivot plot bar no_ablett
        x with_ablett
        y mean win
        by era
        xlabel ""
        ylabel "Win rate"
        title "No Ablett no Geelong?"


# Future additions
# pandas map, qcut, str count, validation of scalar names

# bulk load from folder

# auto-complete after comments...

# Doc versioning...

# check docs, check examples



