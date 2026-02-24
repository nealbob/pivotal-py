# Project: Pivotal

  ## Current focus
 
  - [x] Finalise vscode and jupyter lab extensions

  ## Backlog

  - [ ] I want to Add support for the vs code extention to execute pivotal code. This should include ability to exectue a *.pivotal file via a vs code command / keyboard shortcut. But also I want the ability to execute pivotal code embeded in python files. Fromn our previous discussions this might be possible by putting pivotal code in a notebook cell with the #%% command then putting int the %%pivotal magic command so it can get executed in the interactive notebook enviornment within vs code. I guess this means the vs code interactive notebook needs the jupyter lab extension to be installed. I just tried to do this in cs code but it didnt seem to work (but it did work in jupyer lab standalone running in a browser) perhpas there is something different about the vs code notebook. Could you come up with a plan of required changes here?
5
  ## Ideas (not ready to be implemented)

  - save - I want to have a save option that involves a is more excel like in that the whole session can be saved as a package (like a workbook). So this would mean by default a save command would save all of the tables in the session (each to csvs) in a folder. My preference is to implement this using the Frictionless data standard, so that means saving the session as a frictionless data package, with csvs in folders and matching metadata in json. Meta data needs to be updated each time a command is executed. In future it would be useful to have a version of the metadata in memory so it could be used for autocomplete or AI prompting (should this use a different format in memoery like TOON?). Does it matter that metadata might be duplicative if there are multiple copies of the same columns in different dataframes... Should there be some form of autosave or only save on command...?

  - Audit keywords used in all statements. Where there are multiple keywords pick one and stick with it. Choices should be guided by common keywords in R/dplyr, pandas, SQL, and DAX / power query / excel picking the one that is most obvious to users of these tools (where it is less obvious we can lean towards pandas keywords, e.g., merge rather than join).

  - cast / type conversion — type coercion is fiddly and infrequent. Python is the right place for it.

  - load multiple files or a folder and merge or concat, apply type conversion on load (use json metadata or something to guide this). Perhaps simple load then add settings sub command or, modify metadata then reload using settings in metadata??

  - describe / sample — pure exploration helpers. One-liners in Python (df.describe(), df.sample(10)), not worth adding to the grammar.

  - melt / unpivot — complex, infrequent, and the syntax would be awkward. Python is clearly the right escape hatch.

  - Window / rolling functions — same. The pandas API for these is already fairly readable and they're an advanced use case.

  - head / tail — in a notebook context this is about quick exploration. limit 10 at the end of a pipeline to preview results is very natural and saves a Python cell.

  - connections to other data formats / sources...

## Completed

  

  - [x] Drop columns e.g., drop colA, colB  -> dfA.drop(["colA", "colB"])

  - [x] fillna / dropna — missing value handling is arguably the single most common data cleaning step. Having to drop to Python for this every time would be a constant friction point. These belong in the language.

  - [x] dedupe — keyword is `distinct`, consistent with SQL and R/dplyr

  - [x] concat — combining two tables vertically (e.g. appending monthly CSVs)

  - [x] rename — i.e., rename colA as newcol

  - [x] between / contains in filters — `between [lo, hi]`, `contains`, `not contains`, `startswith`, `endswith`

  - [x] load excel and parquet format data sources — file format auto-detected from suffix (.xlsx, .xls, .parquet, .csv). Works for both literal paths and variable paths (runtime detection).
