# Project: Pivotal

## Current focus

## Backlog


## Ideas

  - [] bug in Jupyterlab pivot GUI need to allow for alias and from 

  - [] bug in load GUi parsing of file paths (fix wiht some pyhton processing)

  - [] graphicwalker in the viewer pane AG grid and gwalker code generation...

    - switch polars to pandas?>?

    - error handling

  - left menu upgrades

    - scatter diagrams

  - describe / sample Visual summary via tabulator? plus showing df changes in left and right pane

  - [] Bugs in plot (y and x) technicaly need to swap for barh...ALSO sharex share y override in plot, rotate x or y text overide, y axis label position?

  - [ ] Export to ms word / excel option to keep formating excel one chart or table per sheet, or dataframe include / exclude data and chart images tables include formating... 

  - [ ] Interactive plots (plotly?)

  - [ ]  Bulk data load. Load multiuple csv files and combine (merge / concat). Load multiple sql tables (combine or don't)

  - [ ] VSCODE extension improvements

  - schema support (read and write) in frictionless / sql

  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤     
  │     │ Database / cloud           │ DuckDB already does the heavy lifting. Exposing it in load syntax turns Pivotal  │
  │ 2   │ connectors (DuckDB)        │ from a local-file tool into something usable on real production data. High ROI,  │   
  │     │                            │ relatively contained.                                                            │
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ 3   │ VS Code interactive output │ VS Code is where most data practitioners work. Without a viewer panel, Pivotal   │   
  │     │                            │ is effectively JupyterLab-only. Biggest audience expansion available.            │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │ 4   │ Data quality assertions    │ Turns scripts into self-validating pipelines. Natural fit for the grammar. Moves │   
  │     │                            │  Pivotal toward production use cases, not just exploratory analysis.             │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │ 5   │ AI / natural language →    │ Pivotal's small grammar makes it a much better LLM target than raw Pandas.       │   
  │     │ Pivotal                    │ Strong differentiation story. Lowers barrier to entry for non-technical users.   │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │ 6   │ Column profiling in        │ Inline dtype, null %, histogram per column. Low effort, high daily utility —     │   
  │     │ explorer                   │ replaces a df.describe() call users currently have to make manually.             │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │     │ More output formats        │ Plotly fills the interactive charts gap. Excel is practical for business users   │   
  │ 7   │ (Plotly first, then Excel) │ handing off results. Both are self-contained additions. Word/docx lowest         │   
  │     │                            │ priority of the three.                                                           │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │     │ Explorer pane enhancements │ Search bar is genuinely useful at scale. Click-to-cast/rename is nice but        │   
  │ 8   │  (search bar, click to     │ secondary — prioritise search first.                                             │   
  │     │ cast/rename)               │                                                                                  │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │     │                            │ Most exciting concept but highest risk. The Vega-Lite → Pivotal translation is   │   
  │ 9   │ Graphic Walker integration │ hard to get right. Better tackled once the core is more mature and after a spike │   
  │     │                            │  to assess feasibility.                                                          │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │     │                            │ High long-term value (editor-agnostic, consolidates autocomplete/error logic)    │   
  │ 10  │ LSP                        │ but pure infrastructure with no visible user-facing payoff until the editors     │   
  │     │                            │ consuming it are built. Revisit after VS Code work is further along.   


## Implementation plans


