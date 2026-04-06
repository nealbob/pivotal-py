# Project: Pivotal

## Current focus

## Backlog

  - Does the agg plot syntax in the pivotal grammer support multiple Y (values) columns? At present the GUI seems to create multiple Y lines

    agg plot mechart
      x colA
      y mean colB
      y mean colC
  
  But only the latter y column is displayed in the chart. To behave more like a regular plot statement it should probably support a comma seperated list

    agg plot mechart
      x colA
      y mean colB, mean colC

  NEed to check if this is ust an issue with the Plot GUI of whether this syntax is actualy supported in the Pivotal language grammer.

include filter in agg plot


  

click on column switches view to that table and puts cursor / focus on first row of that column in the table...

click on view button opens that item in the viewer (even if the viewer window isclosed or viewer = False??)

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

---

### VS Code Extension — Full Feature Parity with JupyterLab

#### Architecture

Two JupyterLab-specific systems need replacing:

**Transport layer** (IPython comm → WebSocket)
```
Python magic ──WebSocket──▶ VS Code Extension ──postMessage──▶ WebviewPanel
             ◀──WebSocket──                   ◀──postMessage──
```

**UI layer** (Lumino widgets → VS Code APIs)
```
JupyterLab Explorer (Lumino Widget)  →  VS Code TreeView (Activity Bar sidebar)
JupyterLab Viewer  (Lumino Widget)   →  VS Code WebviewPanel (editor column right)
JupyterLab Toolbar (Lumino Widget)   →  VS Code commands + keybindings + menus
```

---

#### Phase 1 — Language Support Polish
**Effort: Small (~0.5 day)**

Syntax highlighting for `.pivotal` files is already complete (`syntaxes/pivotal.tmLanguage.json`). Two small additions remain:

- **Hover provider** — show keyword documentation on hover. The `detail` strings are already defined in `COMMAND_COMPLETIONS` in `extension.ts`; wire them up via `vscode.languages.registerHoverProvider`.
- **Snippet Tab-stops** — the snippets in `COMMAND_COMPLETIONS` are already proper VS Code snippet strings (e.g. `group by ${1:grp_col}\n    agg ${2:func}...`) but aren't registered with `InsertTextFormat.Snippet`. Doing so enables Tab-stop navigation through placeholders.

---

#### Phase 2 — Python Communication Layer (WebSocket Bridge)
**Effort: Medium (3–5 days)**

Replaces IPython comm with a local WebSocket server. This is the foundation all subsequent phases depend on.

**Python side (`pivotal/magic.py` or new `pivotal/vscode_bridge.py`):**
- On first send, start an `asyncio` WebSocket server on a random available port (`websockets` library)
- Write `{"port": N, "pid": P}` to a temp file: `os.path.join(tempfile.gettempdir(), 'pivotal_bridge.json')`
- Detect VS Code context via `os.environ.get('VSCODE_PID')` and use the bridge instead of comm
- Message format: **identical JSON to existing comm messages** — `{type, name, data, ...}` — so the viewer JS protocol needs no changes
- Graceful fallback if `websockets` is not installed: disable viewer, show install prompt

**VS Code extension side:**
- Watch temp file for creation/changes using `fs.watch` on activation
- When bridge file appears, read port and open WebSocket connection
- Route incoming messages to viewer and explorer
- Route outgoing messages (row limit requests, deletes) back via WebSocket

**Bridge file approach** means this works transparently whether Python is started via CLI, Jupyter Interactive Window, or any other mechanism.

---

#### Phase 3 — WebView Viewer Panel
**Effort: Large (5–8 days)**

The right-hand data viewer, opened as a split editor panel (`vscode.ViewColumn.Two`).

**VS Code mechanism:**
```typescript
vscode.window.createWebviewPanel(
  'pivotalViewer', 'Pivotal Viewer',
  vscode.ViewColumn.Two,
  { enableScripts: true, retainContextWhenHidden: true }
)
```

**Portability of `viewer.ts` components:**

| Component | Reuse % | Notes |
|---|---|---|
| AG Grid setup, column defs, filters | ~85% | Remove Lumino imports; replace widget attach with HTML |
| `SelectPopupFilter` (categorical) | 100% | Pure AG Grid — no changes |
| Chart rendering (zoom, pan, canvas modes) | ~90% | Replace Lumino signals with `window.addEventListener` |
| GT table iframe rendering | ~90% | Identical — iframe works in webview |
| Navigation history (back/forward) | 100% | No changes |
| Footer (row limit, row count, show all) | ~95% | Replace comm.send with `vscode.postMessage` |
| Clipboard (TSV+HTML, PNG, HTML) | 100% | Clipboard API identical |
| Float formatter, zoom toolbar | 100% | No changes |

**What must be rewritten:**
- `Widget` class hierarchy → flat HTML page with JS
- Panel lifecycle → VS Code `onDidDispose` / `onDidChangeViewState` events
- Communication → `vscode.acquireVsCodeApi().postMessage()` and `window.addEventListener('message', ...)`
- Colours → VS Code CSS variables (`--vscode-editor-background`, `--vscode-foreground`, etc.) fed into AG Grid's CSS variable theming API

**Bundling:** webpack config already exists in the JupyterLab extension — adapt for `editors/vscode/out/`.

---

#### Phase 4 — TreeView Explorer Panel
**Effort: Medium (3–4 days)**

The left-panel object inspector. Unlike the viewer, `explorer.ts` cannot be ported — it must be rewritten using VS Code's `TreeDataProvider` API. Pivotal gets its own icon in the VS Code activity bar.

**`package.json` additions:**
```json
"viewsContainers": {
  "activitybar": [{ "id": "pivotal-explorer", "title": "Pivotal", "icon": "icon.png" }]
},
"views": {
  "pivotal-explorer": [
    { "id": "pivotalData",   "name": "Data"   },
    { "id": "pivotalCharts", "name": "Charts" },
    { "id": "pivotalTables", "name": "Tables" }
  ]
}
```

**Tree structure:**
```
▼ sales  (5000 × 12)
    region   — categorical
    amount   — numeric
    date     — datetime
  revenue_chart
  summary_table
```

- `getChildren()` returns DataFrames/Charts/Tables or column children
- `onDidChangeTreeData` fires on new WebSocket payload
- Click to view → focuses WebviewPanel and renders that item
- Inline action buttons: eye (view), trash (delete)
- Status bar item: `df: sales` — updates on `current_table` message

---

#### Phase 5 — GUI Dialogs
**Effort: Medium (2–3 days)**

JupyterLab's Python-driven widget GUIs replaced with VS Code dialogs that generate and insert Pivotal code. No Python-side GUI code needed.

Simple, linear workflows use **QuickPick/InputBox** sequences. Complex, interactive GUIs use a **WebviewPanel HTML form** (same mechanism as the viewer) so users can see all options at once and iterate quickly — like the JupyterLab widget experience.

**Load Dataset** (`Ctrl+Shift+L`) — QuickPick sequence:
1. `showOpenDialog` — file picker (CSV, XLSX, Parquet)
2. `InputBox` — table name
3. Insert `load <name> "<path>"` at cursor

**Save Package** (`Ctrl+Shift+S`) — QuickPick sequence:
1. `QuickPick` (multi-select) — DataFrames/charts/tables from TreeView
2. `InputBox` — package name
3. `showOpenDialog` (directory) — output path
4. `QuickPick` — format (parquet / csv / xlsx)
5. Insert generated `save` block

**Code Export** (`Ctrl+Shift+E`) — QuickPick sequence:
1. `QuickPick` — backend (pandas / polars / duckdb / sql / pivotal)
2. Run `python -m pivotal --compile --backend <X> "<file>"` → open result

**Plot GUI** — WebviewPanel HTML form (persistent, all controls visible simultaneously):
- Opens as a side panel (or in `ViewColumn.Two` alongside the viewer)
- Form fields: chart type dropdown, X column, Y column, optional group-by column, optional secondary Y, title override
- Columns populated live from the TreeView store (`_explorerItems`)
- "Insert" button generates and inserts the `plot` block at cursor; panel stays open for iteration
- Replicates the JupyterLab experience of flicking between different plot configurations quickly

**Pivot GUI** — WebviewPanel HTML form (persistent, all controls visible simultaneously):
- Form fields: group-by column(s), value column, aggregation function dropdown (sum/mean/count/min/max/wavg), optional alias
- "Insert" button generates and inserts the `group by` / `agg` block at cursor; panel stays open
- Columns populated live from the TreeView store

---

#### Phase 6 — Notebook Integration Polish
**Effort: Small (2–3 days) | Risk: Medium**

- Suppress raw comm output in the interactive window now that the bridge handles display
- Use VS Code Jupyter extension API (`ms-toolsai.jupyter`) to detect kernel state and auto-connect bridge when available — degrade gracefully if not present
- Status bar indicator: bridge connection state (connected / waiting / disconnected)
- Handle kernel restart: watch for WebSocket disconnect, re-watch bridge file, clear explorer

---

#### Risks

| Risk | Mitigation |
|---|---|
| `websockets` not installed in user's Python env | Disable viewer gracefully, show one-time install prompt |
| VS Code Jupyter API changes between versions | Bridge file is primary transport; Jupyter API is enhancement only |
| AG Grid bundle size (~1.5 MB) | Community edition only; tree-shake; load lazily on first render |
| `retainContextWhenHidden` memory cost | Call `gridApi.destroy()` on hide if memory is an issue; reconstruct from cached payload |

---

#### What Does NOT Change

- `magic.py` message format — identical JSON payloads, viewer JS speaks the same protocol
- `pivotal_autocomplete.json` schema — already used by both extensions
- DSL compiler, error handling, Polars/DuckDB backends
- Existing VS Code commands (executeFile, compileToFile, etc.)

---
