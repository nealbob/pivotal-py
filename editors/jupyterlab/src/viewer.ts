import { Widget } from '@lumino/widgets';
import { Message } from '@lumino/messaging';
import {
  createGrid, AllCommunityModule, ModuleRegistry,
  type GridOptions, type ColDef, type GridApi,
  type IFilterComp, type IFilterParams, type IDoesFilterPassParams,
} from 'ag-grid-community';

import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';

// Popup (header ≡ click) filter for categorical/boolean columns.
// Uses a custom div list so hover and selection highlighting work in all browsers.
class SelectPopupFilter implements IFilterComp {
  private _gui!: HTMLElement;
  private _list!: HTMLElement;
  private _value = '';
  private _params!: IFilterParams & { values?: string[] };

  init(params: IFilterParams & { values?: string[] }): void {
    this._params = params;
    const values: string[] = params.values ?? [];

    this._list = document.createElement('div');
    this._list.className = 'pv-filter-list';

    for (const v of values) {
      const item = document.createElement('div');
      item.className = 'pv-filter-option';
      item.dataset['value'] = v;
      item.textContent = v === '' ? 'All' : String(v);
      if (v === '') item.classList.add('pv-filter-selected');  // 'All' highlighted by default
      item.addEventListener('click', () => {
        this._value = v;
        this._list.querySelectorAll('.pv-filter-option').forEach(el =>
          el.classList.toggle('pv-filter-selected', (el as HTMLElement).dataset['value'] === v)
        );
        params.filterChangedCallback();
      });
      this._list.appendChild(item);
    }

    this._gui = document.createElement('div');
    this._gui.appendChild(this._list);
  }

  getGui(): HTMLElement { return this._gui; }

  doesFilterPass(params: IDoesFilterPassParams): boolean {
    if (!this.isFilterActive()) return true;
    const colId = this._params.column.getColId();
    return String((params.data as Record<string, unknown>)[colId] ?? '') === this._value;
  }

  isFilterActive(): boolean { return this._value !== ''; }

  getModel(): { filterType: string; filter: string } | null {
    return this.isFilterActive() ? { filterType: 'text', filter: this._value } : null;
  }

  setModel(model: { filter?: string } | null): void {
    this._value = model?.filter ?? '';
    this._list?.querySelectorAll('.pv-filter-option').forEach(el =>
      el.classList.toggle('pv-filter-selected', (el as HTMLElement).dataset['value'] === this._value)
    );
  }

  destroy(): void { /* nothing to clean up */ }
}

// ---------------------------------------------------------------------------
// Payload types
// ---------------------------------------------------------------------------

export type SemanticColType = 'numeric' | 'categorical' | 'datetime' | 'boolean' | 'string';

export interface DataFramePayload {
  type: 'dataframe';
  name: string;
  columns: string[];
  data: unknown[][];           // row-major: data[rowIdx][colIdx]
  dtypes: Record<string, string>;
  col_types?: Record<string, SemanticColType>;
  shape: [number, number];
  truncated: boolean;
  viewer_font?: number;        // em units for font size (default 0.75)
  viewer_num_format?: number;  // significant digits for floats (0 = off)
}

export interface CanvasMeta {
  page_width_mm: number;
  page_height_mm: number;
  margin_mm: number;
  chart_width_mm?: number;
  chart_height_mm?: number;
  label: string; // e.g. 'A4'
}

export interface ChartPayload {
  type: 'chart';
  name: string;
  data: string; // base64 PNG
  canvas?: CanvasMeta;
  source_df?: string;
}

export interface GtTablePayload {
  type: 'gt_table';
  name: string;
  html: string;
  canvas?: CanvasMeta;
  source_df?: string;
}

export type ViewerMessage = DataFramePayload | ChartPayload | GtTablePayload;

export interface ValueInfo {
  kind: 'scalar' | 'list' | 'dict';
  value_type?: string;
  preview?: string;
  item_type?: string | null;
  length?: number;
  size?: number;
  children?: Record<string, ValueInfo>;
}

export interface ExplorerItem {
  name: string;
  type: 'dataframe' | 'chart' | 'gt_table' | 'value';
  shape?: [number, number];
  columns?: { name: string; dtype: string; col_type?: SemanticColType }[];
  source_df?: string;
  value?: ValueInfo;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_LIMIT = 20_000;
const BASE_ROW_H = 26;   // px at zoom 1.0
const BASE_HDR_H = 30;   // px at zoom 1.0

// ---------------------------------------------------------------------------
// PivotalViewerWidget
//
// Storage: one slot per named object (latest version wins).
// Back/Forward navigates between distinct named objects, not history.
// ---------------------------------------------------------------------------

export class PivotalViewerWidget extends Widget {
  private _latest: Map<string, ViewerMessage> = new Map();
  private _names: string[] = [];
  private _index = -1;
  private _comm: { send(data: unknown): void } | null = null;
  private _contentChangedCb: ((items: ExplorerItem[]) => void) | null = null;
  private _viewingItemCb: ((name: string | null) => void) | null = null;
  private _activateCb: (() => void) | null = null;
  private _zoomCb: ((factor: number) => void) | null = null;

  // Cache AG Grid DOM nodes + GridApi per DataFrame name so back/forward navigation
  // reattaches existing instances rather than rebuilding from scratch.
  private _dfCache: Map<string, { body: HTMLElement; footer: HTMLElement; api: GridApi; applyZoom: (mult: number) => void; resetZoom: () => void }> = new Map();

  // Callback set by canvas renderers (_renderChartOnPage / _renderGtTableOnPage)
  // so that Lumino's onResize can trigger a re-layout when the panel is resized.
  private _panelResizeCb: (() => void) | null = null;
  private _pinMenu: HTMLElement | null = null;

  private _titleEl!: HTMLElement;
  private _counterEl!: HTMLElement;
  private _backBtn!: HTMLButtonElement;
  private _fwdBtn!: HTMLButtonElement;
  private _copyBtn!: HTMLButtonElement;
  private _delBtn!: HTMLButtonElement;
  private _clearBtn!: HTMLButtonElement;
  private _refreshBtn!: HTMLButtonElement;
  private _body!: HTMLElement;
  private _footer!: HTMLElement;

  constructor() {
    super();
    this.addClass('pv-viewer');
    this.id = 'pivotal-viewer-panel';
    this.title.label = '';
    this.title.caption = 'Pivotal Object Viewer';
    this.title.closable = true;

    this.node.innerHTML = `
      <div class="pv-header">
        <div class="pv-nav">
          <button class="pv-btn pv-back" title="Back (Alt+[)">&#9664;</button>
          <button class="pv-btn pv-fwd"  title="Forward (Alt+])">&#9654;</button>
          <span class="pv-title">—</span>
          <span class="pv-counter"></span>
        </div>
        <button class="pv-btn pv-copy"      title="Copy to clipboard">&#128203;</button>
        <button class="pv-btn pv-refresh"   title="Refresh">&#8635;</button>
        <button class="pv-btn pv-del"       title="Delete object">&#10005;</button>
        <button class="pv-btn pv-clear"     title="Delete all">&#128465;</button>
      </div>
      <div class="pv-body"></div>
      <div class="pv-footer"></div>
    `;

    this._titleEl   = this.node.querySelector('.pv-title')  as HTMLElement;
    this._counterEl = this.node.querySelector('.pv-counter') as HTMLElement;
    this._backBtn   = this.node.querySelector('.pv-back')   as HTMLButtonElement;
    this._fwdBtn    = this.node.querySelector('.pv-fwd')    as HTMLButtonElement;
    this._copyBtn   = this.node.querySelector('.pv-copy')   as HTMLButtonElement;
    this._delBtn    = this.node.querySelector('.pv-del')    as HTMLButtonElement;
    this._clearBtn   = this.node.querySelector('.pv-clear')   as HTMLButtonElement;
    this._refreshBtn = this.node.querySelector('.pv-refresh') as HTMLButtonElement;
    this._body       = this.node.querySelector('.pv-body')    as HTMLElement;
    this._footer    = this.node.querySelector('.pv-footer') as HTMLElement;

    this._backBtn.disabled  = true;
    this._fwdBtn.disabled   = true;
    this._copyBtn.disabled  = true;
    this._delBtn.disabled   = true;
    this._clearBtn.disabled   = true;
    this._refreshBtn.disabled = true;

    this.node.tabIndex = -1;

    this._backBtn.addEventListener('click', () => this.back());
    this._fwdBtn.addEventListener('click', () => this.forward());
    this._copyBtn.addEventListener('click', () => this._copyToClipboard());
    this._delBtn.addEventListener('click', () => this.deleteCurrent());
    this._clearBtn.addEventListener('click', () => this.clear());
    this._refreshBtn.addEventListener('click', () => this._refreshCurrent());
  }

  setComm(comm: { send(data: unknown): void }): void {
    this._comm = comm;
  }

  setContentChangedCallback(cb: (items: ExplorerItem[]) => void): void {
    this._contentChangedCb = cb;
  }

  setViewingItemCallback(cb: (name: string | null) => void): void {
    this._viewingItemCb = cb;
  }

  setActivateCallback(cb: () => void): void {
    this._activateCb = cb;
  }

  focusItem(name: string): void {
    const idx = this._names.indexOf(name);
    if (idx >= 0 && idx !== this._index) {
      this._index = idx;
      this._render();
    }
  }

  private _getExplorerItems(): ExplorerItem[] {
    return this._names.map(name => {
      const msg = this._latest.get(name)!;
      if (msg.type === 'dataframe') {
        const df = msg as DataFramePayload;
        return {
          name,
          type: 'dataframe' as const,
          shape: df.shape,
          columns: df.columns.map(c => ({ name: c, dtype: df.dtypes[c] ?? '', col_type: df.col_types?.[c] })),
        };
      }
      if (msg.type === 'chart') return { name, type: 'chart' as const, source_df: (msg as ChartPayload).source_df };
      return { name, type: 'gt_table' as const, source_df: (msg as GtTablePayload).source_df };
    });
  }

  private _notifyContentChanged(): void {
    this._contentChangedCb?.(this._getExplorerItems());
  }

  // -------------------------------------------------------------------------
  // Navigation
  // -------------------------------------------------------------------------

  push(msg: ViewerMessage): void {
    const isNew = !this._latest.has(msg.name);
    this._latest.set(msg.name, msg);
    // Evict stale cache entry so updated data gets a fresh AG Grid instance
    if (!isNew) {
      this._dfCache.get(msg.name)?.api?.destroy();
      this._dfCache.delete(msg.name);
    }
    if (isNew) this._names.push(msg.name);
    this._index = this._names.indexOf(msg.name);
    this._render();
    this._notifyContentChanged();
  }

  back(): void {
    if (this._index > 0) { this._index--; this._render(); }
  }

  forward(): void {
    if (this._index < this._names.length - 1) { this._index++; this._render(); }
  }

  // Remove a named item from viewer without notifying Python (Python initiated this).
  deleteItem(name: string): void {
    const idx = this._names.indexOf(name);
    if (idx < 0) return;
    this._latest.delete(name);
    this._dfCache.get(name)?.api?.destroy();
    this._dfCache.delete(name);
    this._names.splice(idx, 1);
    this._index = Math.min(Math.max(this._index, 0), this._names.length - 1);
    if (this._names.length === 0) {
      this.clear();
    } else {
      if (this._index >= this._names.length) this._index = this._names.length - 1;
      this._render();
      this._notifyContentChanged();
    }
  }

  // Delete current item AND tell Python to delete it from the namespace.
  deleteCurrent(): void {
    if (this._index < 0 || !this._names.length) return;
    const name = this._names[this._index];
    this._comm?.send({ type: 'delete', name });
    this.deleteItem(name);
  }

  // Apply a zoom multiplier to whatever is currently displayed.
  zoom(factor: number): void {
    this._zoomCb?.(factor);
  }

  // Focus the AG Grid so arrow-key navigation works after Alt+V.
  focusGrid(): void {
    const name = this._names[this._index];
    if (name) {
      const cached = this._dfCache.get(name);
      if (cached?.api) {
        try {
          const cols = cached.api.getColumns();
          if (cols && cols.length > 0) {
            cached.api.ensureIndexVisible(0, 'top');
            cached.api.setFocusedCell(0, cols[0]);
            return;
          }
        } catch (_) { /* not a grid view — fall through */ }
      }
    }
    this.node.focus();
  }

  // Scroll the AG Grid for tableName so that colName is visible, then flash it.
  scrollToColumn(tableName: string, colName: string): void {
    const cached = this._dfCache.get(tableName);
    if (!cached?.api) return;
    try {
      cached.api.ensureColumnVisible(colName);
      // Brief highlight on the header cell
      const escaped = CSS.escape(colName);
      const headerEl = this.node.querySelector(`.ag-header-cell[col-id="${escaped}"]`);
      if (headerEl) {
        headerEl.classList.add('pv-col-flash');
        setTimeout(() => headerEl.classList.remove('pv-col-flash'), 800);
      }
    } catch (_) { /* column may not exist */ }
  }

  // Delete a named item AND tell Python to delete it from the namespace.
  requestDelete(name: string): void {
    this._comm?.send({ type: 'delete', name });
    this.deleteItem(name);
  }

  clear(): void {
    this._latest.clear();
    for (const cached of this._dfCache.values()) cached.api?.destroy();
    this._dfCache.clear();
    this._names = [];
    this._index = -1;
    this._titleEl.textContent = '—';
    this._counterEl.textContent = '';
    this._backBtn.disabled = true;
    this._fwdBtn.disabled = true;
    this._copyBtn.disabled = true;
    this._delBtn.disabled = true;
    this._clearBtn.disabled   = true;
    this._refreshBtn.disabled = true;
    this._body.innerHTML = '';
    this._footer.innerHTML = '';
    this._notifyContentChanged();
    this._viewingItemCb?.(null);
  }

  private _refreshCurrent(): void {
    if (this._index < 0 || !this._names.length) return;
    const name = this._names[this._index];
    const inp = this._footer.querySelector('.pv-limit') as HTMLInputElement | null;
    const limit = inp ? Math.max(100, parseInt(inp.value, 10) || DEFAULT_LIMIT) : DEFAULT_LIMIT;
    this._comm?.send({ type: 'request', name, limit });
  }

  // -------------------------------------------------------------------------
  // Clipboard
  // -------------------------------------------------------------------------

  private async _copyToClipboard(): Promise<void> {
    if (this._index < 0 || !this._names.length) return;
    const p = this._latest.get(this._names[this._index]);
    if (!p) return;

    try {
      if (p.type === 'dataframe') {
        // Copy as TSV (pastes into Excel/Sheets) + HTML table (pastes into Word/email)
        const df = p as DataFramePayload;
        const tsv = [
          df.columns.join('\t'),
          ...df.data.map(row => row.map(v => (v == null ? '' : String(v))).join('\t')),
        ].join('\n');
        const html = [
          '<table><thead><tr>',
          df.columns.map(c => `<th>${c}</th>`).join(''),
          '</tr></thead><tbody>',
          ...df.data.map(row =>
            '<tr>' + row.map(v => `<td>${v == null ? '' : String(v)}</td>`).join('') + '</tr>'
          ),
          '</tbody></table>',
        ].join('');
        await navigator.clipboard.write([
          new ClipboardItem({
            'text/plain': new Blob([tsv],  { type: 'text/plain' }),
            'text/html':  new Blob([html], { type: 'text/html'  }),
          }),
        ]);
      } else if (p.type === 'chart') {
        // Copy as PNG image
        const blob = await fetch(`data:image/png;base64,${(p as ChartPayload).data}`).then(r => r.blob());
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      } else {
        // GT table — copy HTML (pastes as formatted table into Word/Outlook)
        const html = (p as GtTablePayload).html;
        await navigator.clipboard.write([
          new ClipboardItem({
            'text/html':  new Blob([html], { type: 'text/html'  }),
            'text/plain': new Blob([html], { type: 'text/plain' }),
          }),
        ]);
      }
      // Brief visual confirmation
      const orig = this._copyBtn.textContent;
      this._copyBtn.textContent = '✓';
      setTimeout(() => { this._copyBtn.textContent = orig; }, 1200);
    } catch (err) {
      console.error('[Pivotal] clipboard copy failed:', err);
      this._copyBtn.title = 'Copy failed — check browser permissions';
    }
  }

  // -------------------------------------------------------------------------
  // Top-level render dispatch
  // -------------------------------------------------------------------------

  private _render(): void {
    if (this._index < 0 || !this._names.length) return;
    const p = this._latest.get(this._names[this._index]);
    if (!p) return;
    this._panelResizeCb = null; // cleared; canvas renderers will re-register if needed
    this._zoomCb = null; // cleared; chart renderers will re-register if needed
    this._viewingItemCb?.(p.name);

    const typeLabel = p.type === 'dataframe' ? 'DataFrame' : p.type === 'chart' ? 'Chart' : 'Table';
    this._titleEl.textContent = `${p.name} · ${typeLabel}`;
    this._counterEl.textContent = `${this._index + 1} / ${this._names.length}`;
    this._backBtn.disabled  = this._index === 0;
    this._fwdBtn.disabled   = this._index === this._names.length - 1;
    this._copyBtn.disabled    = false;
    this._delBtn.disabled     = false;
    this._clearBtn.disabled   = false;
    this._refreshBtn.disabled = p.type !== 'dataframe';

    // Detach children without destroying them — keeps AG Grid instances alive in _dfCache
    while (this._body.firstChild) this._body.removeChild(this._body.firstChild);
    while (this._footer.firstChild) this._footer.removeChild(this._footer.firstChild);

    if (p.type === 'dataframe') this._renderDataFrame(p);
    else if (p.type === 'chart') this._renderChart(p);
    else this._renderGtTable(p as GtTablePayload);
  }

  // -------------------------------------------------------------------------
  // DataFrame — rendered with AG Grid (virtual scroll, sortable, filterable)
  // -------------------------------------------------------------------------

  private _renderDataFrame(p: DataFramePayload): void {
    // Reattach cached nodes if this DataFrame was previously rendered
    const cached = this._dfCache.get(p.name);
    if (cached) {
      this._body.appendChild(cached.body);
      this._footer.appendChild(cached.footer);
      this._zoomCb = cached.applyZoom;
      return;
    }

    const { columns, data, dtypes } = p;
    const sigFigs = p.viewer_num_format ?? 5;

    // Float formatter: sigFigs significant digits, trim trailing zeros, keep sci notation
    const floatFormatter = sigFigs > 0
      ? (params: { value: unknown }) => {
          const val = params.value;
          if (val === null || val === undefined || val === '') return '';
          const n = Number(val);
          if (isNaN(n)) return String(val);
          const s = n.toPrecision(sigFigs);
          return s.includes('e') ? s : String(parseFloat(s));
        }
      : undefined;

    // Convert row-major array to row objects, prepend row index
    const rowData = data.map((row, i) => {
      const obj: Record<string, unknown> = { _idx: i };
      columns.forEach((col, ci) => { obj[col] = row[ci]; });
      return obj;
    });

    const colDefs: ColDef[] = [
      {
        field: '_idx',
        headerName: '',
        pinned: 'left',
        width: 52,
        minWidth: 52,
        maxWidth: 52,
        sortable: false,
        filter: false,
        floatingFilter: false,
        resizable: false,
        suppressMovable: true,
        cellClass: 'pv-ag-idx',
        headerClass: 'pv-ag-idx-header',
      },
      ...columns.map(col => {
        const dt = dtypes[col] ?? '';
        const isFloat = dt.startsWith('float');
        const isNum = isFloat || dt.startsWith('int');
        const semType = (p.col_types ?? {})[col] ?? (isNum ? 'numeric' : 'string');

        const def: ColDef = {
          field: col,
          headerName: col,
          type: isNum ? 'numericColumn' : undefined,
          headerTooltip: dt || undefined,
          resizable: true,
          sortable: true,
        };

        if (semType === 'categorical' || semType === 'boolean') {
          const vals = [...new Set(rowData.map(r => String(r[col] ?? '')))]
            .filter(v => v !== '')
            .sort((a, b) => a.localeCompare(b));
          def.filter = SelectPopupFilter;
          def.filterParams = { values: ['', ...vals] };
        } else if (semType === 'numeric') {
          def.filter = 'agNumberColumnFilter';
        } else {
          // string / datetime — text contains
          def.filter = 'agTextColumnFilter';
          def.filterParams = { filterOptions: ['contains'], defaultOption: 'contains' };
        }

        if (isFloat && floatFormatter) def.valueFormatter = floatFormatter;

        return def;
      }),
    ];

    ModuleRegistry.registerModules([AllCommunityModule]);

    let zoomFactor = p.viewer_font ?? 1.0;

    // Wrapper: zoom toolbar above the AG Grid container
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;flex-direction:column;flex:1;min-height:0;height:100%;';

    const zoomToolbar = document.createElement('div');
    zoomToolbar.className = 'pv-chart-toolbar';
    zoomToolbar.innerHTML = `
      <button class="pv-btn pv-zoom-in"    title="Zoom in (j)">+</button>
      <button class="pv-btn pv-zoom-out"   title="Zoom out (k)">−</button>
      <button class="pv-btn pv-zoom-reset" title="Reset zoom">1:1</button>
      <button class="pv-btn pv-fit-cols"   title="Fit columns to content">↔</button>
    `;

    // viewport clips the scaled container so it always fills the available space
    const viewport = document.createElement('div');
    viewport.style.cssText = 'position:relative; overflow:hidden; flex:1; min-height:0;';

    const container = document.createElement('div');
    container.className = 'pv-ag-container ag-theme-alpine';
    container.style.cssText = 'position:absolute; top:0; left:0; transform-origin:top left;';
    container.style.width  = `${100 / zoomFactor}%`;
    container.style.height = `${100 / zoomFactor}%`;
    container.style.transform = `scale(${zoomFactor})`;

    viewport.appendChild(container);
    wrapper.appendChild(zoomToolbar);
    wrapper.appendChild(viewport);

    // Append to DOM *before* createGrid so container.clientWidth is non-zero
    // when onFirstDataRendered fires (which can happen synchronously).
    this._body.appendChild(wrapper);

    const api = createGrid(container, {
      rowData,
      columnDefs: colDefs,
      rowHeight: BASE_ROW_H,
      headerHeight: BASE_HDR_H,
      defaultColDef: {
        sortable: true,
        filter: true,
        resizable: true,
        minWidth: 60,
        maxWidth: 300,
        width: 120,
      },
      suppressFieldDotNotation: true,   // allow dots in column names (e.g. "2.6")
      animateRows: false,
      enableCellTextSelection: true,
      localeText: { filterOoo: '' },
      onFirstDataRendered: (e) => {
        // Best-effort auto-size after layout settles. The ↔ button is the
        // reliable fallback if the panel was still animating open.
        setTimeout(() => {
          if (container.clientWidth > 0) e.api.autoSizeAllColumns(false);
        }, 600);
      },
    } as GridOptions);

    // Column pin: right-click a header cell to toggle pinned-left
    const hidePinMenu = () => {
      this._pinMenu?.remove();
      this._pinMenu = null;
    };
    container.addEventListener('contextmenu', (e: MouseEvent) => {
      const headerCell = (e.target as Element).closest('.ag-header-cell');
      if (!headerCell) return;
      const colId = headerCell.getAttribute('col-id');
      if (!colId || colId === '_idx') return;
      e.preventDefault();
      hidePinMenu();
      const isPinned = api.getColumnState().find(s => s.colId === colId)?.pinned === 'left';
      const menu = document.createElement('div');
      menu.className = 'pv-pin-menu';
      menu.innerHTML = `<div class="pv-pin-item">${isPinned ? '📌 Unpin column' : '📌 Pin to left'}</div>`;
      menu.style.left = `${e.clientX}px`;
      menu.style.top  = `${e.clientY}px`;
      menu.querySelector('.pv-pin-item')!.addEventListener('click', () => {
        api.applyColumnState({ state: [{ colId, pinned: isPinned ? null : 'left' }] });
        hidePinMenu();
      });
      document.body.appendChild(menu);
      this._pinMenu = menu;
      setTimeout(() => document.addEventListener('click', hidePinMenu, { once: true }), 0);
    });

    const applyZoom = (mult: number) => {
      zoomFactor = Math.max(0.5, Math.min(3.0, zoomFactor * mult));
      container.style.width  = `${100 / zoomFactor}%`;
      container.style.height = `${100 / zoomFactor}%`;
      container.style.transform = `scale(${zoomFactor})`;
    };
    const resetZoom = () => {
      zoomFactor = 1.0;
      container.style.width  = '100%';
      container.style.height = '100%';
      container.style.transform = 'scale(1)';
    };

    zoomToolbar.querySelector('.pv-zoom-in')!   .addEventListener('click', () => applyZoom(1.25));
    zoomToolbar.querySelector('.pv-zoom-out')!  .addEventListener('click', () => applyZoom(1 / 1.25));
    zoomToolbar.querySelector('.pv-zoom-reset')!.addEventListener('click', resetZoom);
    zoomToolbar.querySelector('.pv-fit-cols')!  .addEventListener('click', () => api.autoSizeAllColumns(false));
    this._zoomCb = applyZoom;

    // Footer
    const [totalRows, totalCols] = p.shape;
    const truncMsg = p.truncated
      ? `Showing ${data.length.toLocaleString()} of ${totalRows.toLocaleString()} rows`
      : `${totalRows.toLocaleString()} rows × ${totalCols} cols`;

    const footer = document.createElement('div');
    footer.className = 'pv-footer-bar';
    footer.innerHTML = `
      <span class="pv-shape">${truncMsg}</span>
      ${p.truncated
        ? `<label class="pv-limit-label">Show:
            <input class="pv-limit" type="number" value="${data.length}" min="100" step="1000">
            rows</label>
           <button class="pv-btn pv-show-all" title="Load all rows">Show all</button>`
        : ''}
    `;

    if (p.truncated) {
      const inp = footer.querySelector('.pv-limit') as HTMLInputElement;
      inp.addEventListener('change', () => {
        const limit = Math.max(100, parseInt(inp.value, 10) || DEFAULT_LIMIT);
        if (this._comm) this._comm.send({ type: 'request', name: p.name, limit });
      });
      const showAllBtn = footer.querySelector('.pv-show-all') as HTMLButtonElement;
      showAllBtn.addEventListener('click', () => {
        if (this._comm) this._comm.send({ type: 'request', name: p.name, limit: p.shape[0] });
      });
    }

    // Cache the entry (wrapper already appended above, before createGrid)
    this._dfCache.set(p.name, { body: wrapper, footer, api, applyZoom, resetZoom });
    this._footer.appendChild(footer);
  }

  // -------------------------------------------------------------------------
  // Chart — dispatch to page-layout or free zoom/pan depending on payload
  // -------------------------------------------------------------------------

  private _renderChart(p: ChartPayload): void {
    if (p.canvas) {
      this._renderChartOnPage(p);
    } else {
      this._renderChartFree(p);
    }
  }

  // Free-form zoom/pan (no canvas defined)
  private _renderChartFree(p: ChartPayload): void {
    let scale = 1.0;
    let dragging = false;
    let dragStartX = 0, dragStartY = 0, scrollStartX = 0, scrollStartY = 0;

    const toolbar = document.createElement('div');
    toolbar.className = 'pv-chart-toolbar';
    toolbar.innerHTML = `
      <button class="pv-btn pv-zoom-in"    title="Zoom in">+</button>
      <button class="pv-btn pv-zoom-out"   title="Zoom out">−</button>
      <button class="pv-btn pv-zoom-reset" title="Reset zoom">1:1</button>
    `;

    const scroll = document.createElement('div');
    scroll.className = 'pv-chart-scroll';

    const img = document.createElement('img');
    img.className = 'pv-chart-img';
    img.src = `data:image/png;base64,${p.data}`;
    img.draggable = false;

    scroll.appendChild(img);
    this._body.appendChild(toolbar);
    this._body.appendChild(scroll);

    const setScale = (s: number) => {
      scale = Math.max(0.2, Math.min(5, s));
      img.style.transform = `scale(${scale})`;
      img.style.transformOrigin = 'top left';
    };

    this._zoomCb = (f: number) => setScale(scale * f);

    toolbar.querySelector('.pv-zoom-in')!   .addEventListener('click', () => setScale(scale * 1.25));
    toolbar.querySelector('.pv-zoom-out')!  .addEventListener('click', () => setScale(scale / 1.25));
    toolbar.querySelector('.pv-zoom-reset')!.addEventListener('click', () => setScale(1));

    scroll.addEventListener('pointerdown', (e: PointerEvent) => {
      if (e.button !== 0) return;
      dragging = true;
      dragStartX = e.clientX; dragStartY = e.clientY;
      scrollStartX = scroll.scrollLeft; scrollStartY = scroll.scrollTop;
      scroll.setPointerCapture(e.pointerId);
      scroll.style.cursor = 'grabbing';
    });
    scroll.addEventListener('pointermove', (e: PointerEvent) => {
      if (!dragging) return;
      scroll.scrollLeft = scrollStartX - (e.clientX - dragStartX);
      scroll.scrollTop  = scrollStartY - (e.clientY - dragStartY);
    });
    scroll.addEventListener('pointerup', () => {
      dragging = false;
      scroll.style.cursor = 'grab';
    });
  }

  // Page-layout view: chart placed on a to-scale page background
  private _renderChartOnPage(p: ChartPayload): void {
    const cm = p.canvas!;
    let userScale = 1.0;

    const toolbar = document.createElement('div');
    toolbar.className = 'pv-chart-toolbar';
    toolbar.innerHTML = `
      <button class="pv-btn pv-zoom-in"    title="Zoom in">+</button>
      <button class="pv-btn pv-zoom-out"   title="Zoom out">−</button>
      <button class="pv-btn pv-zoom-reset" title="Fit to panel">Fit</button>
      <span class="pv-canvas-label">${cm.label} · ${cm.margin_mm}mm margins</span>
    `;

    const outer = document.createElement('div');
    outer.className = 'pv-page-view';

    const page = document.createElement('div');
    page.className = 'pv-page';

    const img = document.createElement('img');
    img.src = `data:image/png;base64,${p.data}`;
    img.className = 'pv-page-chart-img';
    img.draggable = false;

    page.appendChild(img);
    outer.appendChild(page);
    this._body.appendChild(toolbar);
    this._body.appendChild(outer);

    // Track last layout inputs so we can skip no-op repaints and break
    // ResizeObserver feedback loops (scrollbar appearing/disappearing cycles).
    let lastAvailW = -1;
    let rafId = 0;

    const apply = () => {
      // Base scale: fit page width into available panel width (32px padding each side)
      const availW = Math.max(outer.clientWidth - 64, 80);
      // Skip if width unchanged — prevents scrollbar-triggered oscillation
      if (Math.abs(availW - lastAvailW) < 1 && rafId === 0) return;
      lastAvailW = availW;
      rafId = 0;

      const pxPerMm = (availW / cm.page_width_mm) * userScale;

      page.style.width  = `${cm.page_width_mm  * pxPerMm}px`;
      page.style.height = `${cm.page_height_mm * pxPerMm}px`;

      img.style.width  = `${(cm.chart_width_mm  ?? cm.page_width_mm  - 2 * cm.margin_mm) * pxPerMm}px`;
      img.style.height = `${(cm.chart_height_mm ?? cm.page_height_mm - 2 * cm.margin_mm) * pxPerMm}px`;
      img.style.left   = `${cm.margin_mm * pxPerMm}px`;
      img.style.top    = `${cm.margin_mm * pxPerMm}px`;
    };

    // Zoom buttons bypass the width-change guard since userScale changed
    const applyForced = () => { lastAvailW = -1; apply(); };

    this._panelResizeCb = applyForced;
    this._zoomCb = (f: number) => { userScale *= f; applyForced(); };

    toolbar.querySelector('.pv-zoom-in')!   .addEventListener('click', () => { userScale *= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-out')!  .addEventListener('click', () => { userScale /= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-reset')!.addEventListener('click', () => { userScale = 1.0;   applyForced(); });

    // Coalesce rapid ResizeObserver callbacks into one RAF to prevent loops
    new ResizeObserver(() => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(apply);
    }).observe(outer);
    requestAnimationFrame(apply);
  }

  // ---------------------------------------------------------------------------
  // GT Table rendering
  // ---------------------------------------------------------------------------

  private _renderGtTable(p: GtTablePayload): void {
    if (p.canvas) {
      this._renderGtTableOnPage(p);
    } else {
      this._renderGtTableFree(p);
    }
  }

  private _renderGtTableFree(p: GtTablePayload): void {
    const iframe = document.createElement('iframe');
    iframe.srcdoc = p.html;
    iframe.setAttribute('sandbox', 'allow-same-origin allow-scripts');
    iframe.style.cssText = 'flex:1; width:100%; height:100%; border:none;';
    this._body.appendChild(iframe);
  }

  private _renderGtTableOnPage(p: GtTablePayload): void {
    const cm = p.canvas!;
    let userScale = 1.0;

    const usableW_mm = cm.page_width_mm - 2 * cm.margin_mm;

    const toolbar = document.createElement('div');
    toolbar.className = 'pv-chart-toolbar';
    toolbar.innerHTML = `
      <button class="pv-btn pv-zoom-in"    title="Zoom in">+</button>
      <button class="pv-btn pv-zoom-out"   title="Zoom out">−</button>
      <button class="pv-btn pv-zoom-reset" title="Fit to panel">Fit</button>
      <span class="pv-canvas-label">${cm.label} · ${cm.margin_mm}mm margins</span>
    `;

    const outer = document.createElement('div');
    outer.className = 'pv-page-view';

    const page = document.createElement('div');
    page.className = 'pv-page';

    // GT HTML column widths may be fixed or flexible. Load the iframe at the
    // CSS equivalent of the usable canvas width so flexible tables render at
    // their natural A4-width layout (not squeezed into the browser's 300px
    // iframe default). After load we measure scrollWidth to get the true
    // content width, then lock and scale from there.
    // allow-scripts: needed for great_tables JS (tooltips etc.)
    // allow-same-origin: needed to read contentDocument dimensions after load
    const CSS_PX_PER_MM = 96 / 25.4;
    const loadW = Math.round(usableW_mm * CSS_PX_PER_MM); // e.g. ~601px for A4
    const iframe = document.createElement('iframe');
    iframe.srcdoc = p.html;
    iframe.setAttribute('sandbox', 'allow-same-origin allow-scripts');
    iframe.style.cssText = `position:absolute; border:none; visibility:hidden; width:${loadW}px;`;

    page.appendChild(iframe);
    outer.appendChild(page);
    this._body.appendChild(toolbar);
    this._body.appendChild(outer);

    let naturalW = 0;   // GT content's natural pixel width (set on iframe load)
    let naturalH = 0;
    let lastAvailW = -1;
    let rafId = 0;

    const apply = () => {
      if (naturalW === 0) return; // iframe not yet loaded

      const availW = Math.max(outer.clientWidth - 64, 80);
      if (Math.abs(availW - lastAvailW) < 1 && rafId === 0) return;
      lastAvailW = availW;
      rafId = 0;

      const pxPerMm  = (availW / cm.page_width_mm) * userScale;
      const marginPx = cm.margin_mm * pxPerMm;

      // Convert the table's natural CSS-pixel width to mm (at 96 DPI).
      // If the table is narrower than the usable area, render it at its true
      // physical size (don't stretch it). If wider, scale it down to fit.
      const naturalW_mm = naturalW / CSS_PX_PER_MM;
      const targetW_mm  = Math.min(naturalW_mm, usableW_mm);
      const scale       = (targetW_mm * pxPerMm) / naturalW;

      page.style.width  = `${cm.page_width_mm  * pxPerMm}px`;
      page.style.height = `${cm.page_height_mm * pxPerMm}px`;

      iframe.style.left            = `${marginPx}px`;
      iframe.style.top             = `${marginPx}px`;
      iframe.style.transform       = `scale(${scale})`;
      iframe.style.transformOrigin = '0 0';
    };

    const applyForced = () => { lastAvailW = -1; apply(); };
    this._panelResizeCb = applyForced;
    this._zoomCb = (f: number) => { userScale *= f; applyForced(); };

    iframe.addEventListener('load', () => {
      try {
        const doc = iframe.contentDocument!;
        naturalW = Math.max(doc.documentElement.scrollWidth, doc.body?.scrollWidth ?? 0, 1);
        naturalH = Math.max(doc.documentElement.scrollHeight, doc.body?.scrollHeight ?? 0, 1);
      } catch {
        // Fallback if same-origin access fails (shouldn't happen with srcdoc)
        naturalW = 800;
        naturalH = 600;
      }
      iframe.style.width      = `${naturalW}px`;
      iframe.style.height     = `${naturalH}px`;
      iframe.style.visibility = '';
      applyForced();
    });

    toolbar.querySelector('.pv-zoom-in')!   .addEventListener('click', () => { userScale *= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-out')!  .addEventListener('click', () => { userScale /= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-reset')!.addEventListener('click', () => { userScale = 1.0;   applyForced(); });

    new ResizeObserver(() => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(apply);
    }).observe(outer);
  }

  protected override onActivateRequest(msg: Message): void {
    super.onActivateRequest(msg);
    this._activateCb?.();
  }

  protected override onResize(_msg: Message): void {
    // Lumino sends this when the panel is resized by the splitter.
    // Canvas renderers register a callback so they can re-scale their content.
    this._panelResizeCb?.();
  }
}
