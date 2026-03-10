import { Widget } from '@lumino/widgets';
import { Message } from '@lumino/messaging';

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
}

export interface CanvasMeta {
  page_width_mm: number;
  page_height_mm: number;
  margin_mm: number;
  chart_width_mm: number;
  chart_height_mm: number;
  label: string; // e.g. 'A4'
}

export interface ChartPayload {
  type: 'chart';
  name: string;
  data: string; // base64 PNG
  canvas?: CanvasMeta;
}

export type ViewerMessage = DataFramePayload | ChartPayload;

export interface ExplorerItem {
  name: string;
  type: 'dataframe' | 'chart';
  shape?: [number, number];
  columns?: { name: string; dtype: string }[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_LIMIT = 10_000;
const ROW_H = 24;     // px — assumed fixed row height for virtual scrolling
const OVERSCAN = 20;  // rows to render beyond the visible viewport edge

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
    this.title.label = 'Pivotal Viewer';
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
      return { name, type: 'chart' as const };
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

    this._titleEl.textContent   = `${p.name} · ${p.type === 'dataframe' ? 'DataFrame' : 'Chart'}`;
    this._counterEl.textContent = `${this._index + 1} / ${this._names.length}`;
    this._backBtn.disabled  = this._index === 0;
    this._fwdBtn.disabled   = this._index === this._names.length - 1;
    this._delBtn.disabled   = false;
    this._clearBtn.disabled = false;

    this._body.innerHTML   = '';
    this._footer.innerHTML = '';

    p.type === 'dataframe' ? this._renderDataFrame(p) : this._renderChart(p);
  }

  // -------------------------------------------------------------------------
  // DataFrame — virtual scrolling via spacer-row technique
  //
  // Single <table> so thead and tbody always scroll together horizontally.
  // thead is position:sticky so it stays visible during vertical scroll.
  // tbody contains: [top-spacer TR] [visible rows] [bottom-spacer TR]
  // Spacer heights are adjusted on scroll to simulate the full row list.
  // -------------------------------------------------------------------------

  private _renderDataFrame(p: DataFramePayload): void {
    const { columns, data } = p;
    const totalRows = data.length;

    // Which column indices are numeric
    const numericCols = new Set(
      columns.reduce<number[]>((acc, c, i) => {
        const dt = p.dtypes[c] ?? '';
        if (dt.startsWith('float') || dt.startsWith('int')) acc.push(i);
        return acc;
      }, [])
    );

    // Scroll container
    const scroll = document.createElement('div');
    scroll.className = 'pv-table-scroll';

    // Single table: thead sticky, tbody has spacers + visible rows
    const table = document.createElement('table');
    table.className = 'pv-table';

    // --- thead ---
    const thead = table.createTHead();
    const headerRow = thead.insertRow();
    const thIdx = document.createElement('th');
    thIdx.className = 'pv-idx';
    headerRow.appendChild(thIdx);
    columns.forEach((col, ci) => {
      const th = document.createElement('th');
      th.textContent = col;
      th.title = p.dtypes[col] ?? '';
      if (numericCols.has(ci)) th.classList.add('pv-num');
      headerRow.appendChild(th);
    });

    // --- tbody with spacers ---
    const tbody = table.createTBody();
    // Top spacer
    const topSpacer = document.createElement('tr');
    const topTd = document.createElement('td');
    topTd.colSpan = columns.length + 1;
    topTd.style.padding = '0';
    topSpacer.appendChild(topTd);
    topSpacer.style.height = '0px';
    tbody.appendChild(topSpacer);
    // Bottom spacer
    const botSpacer = document.createElement('tr');
    const botTd = document.createElement('td');
    botTd.colSpan = columns.length + 1;
    botTd.style.padding = '0';
    botSpacer.appendChild(botTd);
    botSpacer.style.height = `${totalRows * ROW_H}px`;
    tbody.appendChild(botSpacer);

    scroll.appendChild(table);
    this._body.appendChild(scroll);

    // --- Virtual scroll update ---
    let lastStart = -1;

    const update = () => {
      const headerH = thead.offsetHeight || ROW_H;
      const scrollTop = Math.max(0, scroll.scrollTop - headerH);
      const viewH    = scroll.clientHeight - headerH;

      const startIdx = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
      const endIdx   = Math.min(totalRows, Math.ceil((scrollTop + viewH) / ROW_H) + OVERSCAN);

      if (startIdx === lastStart) return;
      lastStart = startIdx;

      // Remove all rows between the two spacers
      while (tbody.rows.length > 2) tbody.deleteRow(1);

      // Adjust spacer heights
      topSpacer.style.height = `${startIdx * ROW_H}px`;
      botSpacer.style.height = `${(totalRows - endIdx) * ROW_H}px`;

      // Build visible rows into a fragment, then insert before bottom spacer
      const frag = document.createDocumentFragment();
      for (let i = startIdx; i < endIdx; i++) {
        const row = document.createElement('tr');
        const idxCell = document.createElement('td');
        idxCell.className = 'pv-idx';
        idxCell.textContent = String(i);
        row.appendChild(idxCell);
        for (let ci = 0; ci < columns.length; ci++) {
          const cell = document.createElement('td');
          const val = data[i][ci];
          cell.textContent = (val === null || val === undefined) ? '' : String(val);
          if (numericCols.has(ci)) cell.classList.add('pv-num');
          row.appendChild(cell);
        }
        frag.appendChild(row);
      }
      tbody.insertBefore(frag, botSpacer);
    };

    scroll.addEventListener('scroll', update, { passive: true });
    new ResizeObserver(update).observe(scroll);
    requestAnimationFrame(update);

    // Footer
    const [totalShape, totalCols] = p.shape;
    const truncMsg = p.truncated
      ? `Showing ${totalRows.toLocaleString()} of ${totalShape.toLocaleString()} rows`
      : `${totalShape.toLocaleString()} rows × ${totalCols} cols`;

    const footer = document.createElement('div');
    footer.className = 'pv-footer-bar';
    footer.innerHTML = `
      <span class="pv-shape">${truncMsg}</span>
      ${p.truncated
        ? `<label class="pv-limit-label">Show:
            <input class="pv-limit" type="number" value="${totalRows}" min="100" step="1000">
            rows</label>`
        : ''}
    `;
    this._footer.appendChild(footer);

    if (p.truncated) {
      const inp = this._footer.querySelector('.pv-limit') as HTMLInputElement;
      inp.addEventListener('change', () => {
        const limit = Math.max(100, parseInt(inp.value, 10) || DEFAULT_LIMIT);
        if (this._comm) this._comm.send({ type: 'request', name: p.name, limit });
      });
    }
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

      img.style.width  = `${cm.chart_width_mm  * pxPerMm}px`;
      img.style.height = `${cm.chart_height_mm * pxPerMm}px`;
      img.style.left   = `${cm.margin_mm * pxPerMm}px`;
      img.style.top    = `${cm.margin_mm * pxPerMm}px`;
    };

    // Zoom buttons bypass the width-change guard since userScale changed
    const applyForced = () => { lastAvailW = -1; apply(); };

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

  protected override onActivateRequest(msg: Message): void {
    super.onActivateRequest(msg);
    this._activateCb?.();
  }

  protected onResize(_msg: Message): void { /* flexbox handles layout */ }
}
