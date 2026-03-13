import { Widget } from '@lumino/widgets';
import { Message } from '@lumino/messaging';
import { TabulatorFull as Tabulator, type ColumnDefinition } from 'tabulator-tables';
import 'tabulator-tables/dist/css/tabulator_simple.min.css';

// ---------------------------------------------------------------------------
// Payload types
// ---------------------------------------------------------------------------

export interface DataFramePayload {
  type: 'dataframe';
  name: string;
  columns: string[];
  data: unknown[][];           // row-major: data[rowIdx][colIdx]
  dtypes: Record<string, string>;
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
}

export interface GtTablePayload {
  type: 'gt_table';
  name: string;
  html: string;
  canvas?: CanvasMeta;
}

export type ViewerMessage = DataFramePayload | ChartPayload | GtTablePayload;

export interface ExplorerItem {
  name: string;
  type: 'dataframe' | 'chart' | 'gt_table';
  shape?: [number, number];
  columns?: { name: string; dtype: string }[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_LIMIT = 10_000;

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
  private _activateCb: (() => void) | null = null;

  // Cache live Tabulator DOM nodes per DataFrame name so back/forward navigation
  // reattaches existing instances rather than rebuilding from scratch.
  private _dfCache: Map<string, { body: HTMLElement; footer: HTMLElement }> = new Map();

  // Callback set by canvas renderers (_renderChartOnPage / _renderGtTableOnPage)
  // so that Lumino's onResize can trigger a re-layout when the panel is resized.
  private _panelResizeCb: (() => void) | null = null;

  private _titleEl!: HTMLElement;
  private _counterEl!: HTMLElement;
  private _backBtn!: HTMLButtonElement;
  private _fwdBtn!: HTMLButtonElement;
  private _delBtn!: HTMLButtonElement;
  private _clearBtn!: HTMLButtonElement;
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
        <button class="pv-btn pv-back"  title="Back (Alt+[)">&#9664;</button>
        <span class="pv-title">—</span>
        <span class="pv-counter"></span>
        <button class="pv-btn pv-fwd"   title="Forward (Alt+])">&#9654;</button>
        <button class="pv-btn pv-del"   title="Delete current">&#10005;</button>
        <button class="pv-btn pv-clear" title="Clear all">&#128465;</button>
      </div>
      <div class="pv-body"></div>
      <div class="pv-footer"></div>
    `;

    this._titleEl   = this.node.querySelector('.pv-title')  as HTMLElement;
    this._counterEl = this.node.querySelector('.pv-counter') as HTMLElement;
    this._backBtn   = this.node.querySelector('.pv-back')   as HTMLButtonElement;
    this._fwdBtn    = this.node.querySelector('.pv-fwd')    as HTMLButtonElement;
    this._delBtn    = this.node.querySelector('.pv-del')    as HTMLButtonElement;
    this._clearBtn  = this.node.querySelector('.pv-clear')  as HTMLButtonElement;
    this._body      = this.node.querySelector('.pv-body')   as HTMLElement;
    this._footer    = this.node.querySelector('.pv-footer') as HTMLElement;

    this._backBtn.disabled  = true;
    this._fwdBtn.disabled   = true;
    this._delBtn.disabled   = true;
    this._clearBtn.disabled = true;

    this._backBtn.addEventListener('click', () => this.back());
    this._fwdBtn.addEventListener('click', () => this.forward());
    this._delBtn.addEventListener('click', () => this.deleteCurrent());
    this._clearBtn.addEventListener('click', () => this.clear());
  }

  setComm(comm: { send(data: unknown): void }): void {
    this._comm = comm;
  }

  setContentChangedCallback(cb: (items: ExplorerItem[]) => void): void {
    this._contentChangedCb = cb;
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
          columns: df.columns.map(c => ({ name: c, dtype: df.dtypes[c] ?? '' })),
        };
      }
      if (msg.type === 'chart') return { name, type: 'chart' as const };
      return { name, type: 'gt_table' as const };
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
    // Evict stale cache entry so updated data gets a fresh Tabulator instance
    if (!isNew) this._dfCache.delete(msg.name);
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

  deleteCurrent(): void {
    if (this._index < 0 || !this._names.length) return;
    const name = this._names[this._index];
    this._latest.delete(name);
    this._dfCache.delete(name);
    this._names.splice(this._index, 1);
    // Move index to the previous item, or stay at 0
    this._index = Math.min(this._index, this._names.length - 1);
    if (this._names.length === 0) {
      this.clear();
    } else {
      this._render();
      this._notifyContentChanged();
    }
  }

  clear(): void {
    this._latest.clear();
    this._dfCache.clear();
    this._names = [];
    this._index = -1;
    this._titleEl.textContent = '—';
    this._counterEl.textContent = '';
    this._backBtn.disabled = true;
    this._fwdBtn.disabled = true;
    this._delBtn.disabled = true;
    this._clearBtn.disabled = true;
    this._body.innerHTML = '';
    this._footer.innerHTML = '';
    this._notifyContentChanged();
  }

  // -------------------------------------------------------------------------
  // Top-level render dispatch
  // -------------------------------------------------------------------------

  private _render(): void {
    if (this._index < 0 || !this._names.length) return;
    const p = this._latest.get(this._names[this._index]);
    if (!p) return;
    this._panelResizeCb = null; // cleared; canvas renderers will re-register if needed

    const typeLabel = p.type === 'dataframe' ? 'DataFrame' : p.type === 'chart' ? 'Chart' : 'Table';
    this._titleEl.textContent = `${p.name} · ${typeLabel}`;
    this._counterEl.textContent = `${this._index + 1} / ${this._names.length}`;
    this._backBtn.disabled  = this._index === 0;
    this._fwdBtn.disabled   = this._index === this._names.length - 1;
    this._delBtn.disabled   = false;
    this._clearBtn.disabled = false;

    // Detach children without destroying them — keeps Tabulator instances alive in _dfCache
    while (this._body.firstChild) this._body.removeChild(this._body.firstChild);
    while (this._footer.firstChild) this._footer.removeChild(this._footer.firstChild);

    if (p.type === 'dataframe') this._renderDataFrame(p);
    else if (p.type === 'chart') this._renderChart(p);
    else this._renderGtTable(p as GtTablePayload);
  }

  // -------------------------------------------------------------------------
  // DataFrame — rendered with Tabulator (virtual DOM, sortable columns)
  // -------------------------------------------------------------------------

  private _renderDataFrame(p: DataFramePayload): void {
    // Reattach cached nodes if this DataFrame was previously rendered
    const cached = this._dfCache.get(p.name);
    if (cached) {
      this._body.appendChild(cached.body);
      this._footer.appendChild(cached.footer);
      return;
    }

    const { columns, data, dtypes } = p;
    const sigFigs = p.viewer_num_format ?? 5;

    // Float formatter: sigFigs significant digits, trim trailing zeros, keep sci notation
    const floatFormatter = sigFigs > 0
      ? (cell: { getValue(): unknown }) => {
          const val = cell.getValue();
          if (val === null || val === undefined || val === '') return '';
          const n = Number(val);
          if (isNaN(n)) return String(val);
          const s = n.toPrecision(sigFigs);
          return s.includes('e') ? s : String(parseFloat(s));
        }
      : undefined;

    // Convert row-major array to Tabulator row objects, prepend row index
    const rows = data.map((row, i) => {
      const obj: Record<string, unknown> = { _idx: i };
      columns.forEach((col, ci) => { obj[col] = row[ci]; });
      return obj;
    });

    const colDefs: ColumnDefinition[] = [
      {
        title: '', field: '_idx',
        frozen: true, width: 52, minWidth: 52,
        hozAlign: 'right', headerSort: false, resizable: false,
        cssClass: 'pv-tab-idx',
      },
      ...columns.map(col => {
        const dt = dtypes[col] ?? '';
        const isFloat = dt.startsWith('float');
        const isNum = isFloat || dt.startsWith('int');
        const colDef: ColumnDefinition = {
          title: col, field: col,
          hozAlign: (isNum ? 'right' : 'left') as 'right' | 'left',
          sorter: (isNum ? 'number' : 'string') as 'number' | 'string',
          tooltip: (dt || false) as string | false,
          resizable: true,
        };
        if (isFloat && floatFormatter) colDef.formatter = floatFormatter as never;
        return colDef;
      }),
    ];

    const container = document.createElement('div');
    container.className = 'pv-tab-container';
    container.style.fontSize = `${p.viewer_font ?? 1.0}em`;

    new Tabulator(container, {
      data: rows,
      columns: colDefs,
      layout: 'fitData',
      height: '100%',
      renderVertical: 'virtual',
      rowHeight: 24,
      nestedFieldSeparator: false,  // allow dots in column names (e.g. "2.6")
    });

    // Footer
    const [totalShape, totalCols] = p.shape;
    const truncMsg = p.truncated
      ? `Showing ${data.length.toLocaleString()} of ${totalShape.toLocaleString()} rows`
      : `${totalShape.toLocaleString()} rows × ${totalCols} cols`;

    const footer = document.createElement('div');
    footer.className = 'pv-footer-bar';
    footer.innerHTML = `
      <span class="pv-shape">${truncMsg}</span>
      ${p.truncated
        ? `<label class="pv-limit-label">Show:
            <input class="pv-limit" type="number" value="${data.length}" min="100" step="1000">
            rows</label>`
        : ''}
    `;

    if (p.truncated) {
      const inp = footer.querySelector('.pv-limit') as HTMLInputElement;
      inp.addEventListener('change', () => {
        const limit = Math.max(100, parseInt(inp.value, 10) || DEFAULT_LIMIT);
        if (this._comm) this._comm.send({ type: 'request', name: p.name, limit });
      });
    }

    // Cache nodes before appending so back/forward reuses live Tabulator instance
    this._dfCache.set(p.name, { body: container, footer });
    this._body.appendChild(container);
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
    iframe.setAttribute('sandbox', 'allow-same-origin');
    iframe.style.cssText = 'flex:1; width:100%; height:100%; border:none;';
    this._body.appendChild(iframe);
  }

  private _renderGtTableOnPage(p: GtTablePayload): void {
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

    const iframe = document.createElement('iframe');
    iframe.srcdoc = p.html;
    iframe.setAttribute('sandbox', 'allow-same-origin');
    iframe.style.cssText = 'position:absolute; border:none;';

    page.appendChild(iframe);
    outer.appendChild(page);
    this._body.appendChild(toolbar);
    this._body.appendChild(outer);

    // GT HTML has hardcoded px column widths and doesn't reflow when the iframe
    // is resized. Fix iframe dimensions on the first render; all subsequent zoom
    // and pane-resize changes go through transform: scale() so canvas and table
    // always stay in sync.
    let initialBasePxPerMm = 0;
    let baseUsableW = 0;
    let baseUsableH = 0;

    let lastAvailW = -1;
    let rafId = 0;

    const apply = () => {
      const availW = Math.max(outer.clientWidth - 64, 80);
      const currentBasePxPerMm = availW / cm.page_width_mm;

      // Lock iframe dimensions on first render
      if (initialBasePxPerMm === 0) {
        initialBasePxPerMm = currentBasePxPerMm;
        baseUsableW = (cm.page_width_mm  - 2 * cm.margin_mm) * initialBasePxPerMm;
        baseUsableH = (cm.page_height_mm - 2 * cm.margin_mm) * initialBasePxPerMm;
        iframe.style.width  = `${baseUsableW}px`;
        iframe.style.height = `${baseUsableH}px`;
      }

      if (Math.abs(availW - lastAvailW) < 1 && rafId === 0) return;
      lastAvailW = availW;
      rafId = 0;

      const pxPerMm  = currentBasePxPerMm * userScale;
      const marginPx = cm.margin_mm * pxPerMm;

      page.style.width  = `${cm.page_width_mm  * pxPerMm}px`;
      page.style.height = `${cm.page_height_mm * pxPerMm}px`;

      iframe.style.left            = `${marginPx}px`;
      iframe.style.top             = `${marginPx}px`;
      // Combine pane-resize ratio and user zoom into one scale — iframe dimensions stay fixed
      iframe.style.transform       = `scale(${(currentBasePxPerMm / initialBasePxPerMm) * userScale})`;
      iframe.style.transformOrigin = '0 0';
    };

    const applyForced = () => { lastAvailW = -1; apply(); };

    this._panelResizeCb = applyForced;

    toolbar.querySelector('.pv-zoom-in')!   .addEventListener('click', () => { userScale *= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-out')!  .addEventListener('click', () => { userScale /= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-reset')!.addEventListener('click', () => { userScale = 1.0;   applyForced(); });

    new ResizeObserver(() => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(apply);
    }).observe(outer);
    requestAnimationFrame(apply);
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
