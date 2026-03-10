import { Widget } from '@lumino/widgets';
import { ExplorerItem } from './viewer';

// ---------------------------------------------------------------------------
// Inline SVG icons (14×14, currentColor)
// ---------------------------------------------------------------------------

const DF_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <rect x="1" y="1" width="12" height="3" rx="0.5" fill="currentColor" opacity="0.75"/>
  <rect x="1" y="5.5" width="5" height="2.2" rx="0.4" fill="currentColor" opacity="0.55"/>
  <rect x="8" y="5.5" width="5" height="2.2" rx="0.4" fill="currentColor" opacity="0.55"/>
  <rect x="1" y="9.3" width="5" height="2.2" rx="0.4" fill="currentColor" opacity="0.4"/>
  <rect x="8" y="9.3" width="5" height="2.2" rx="0.4" fill="currentColor" opacity="0.4"/>
</svg>`;

const CHART_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <rect x="1"   y="7"  width="2.5" height="6" rx="0.4" fill="currentColor" opacity="0.6"/>
  <rect x="5"   y="3"  width="2.5" height="10" rx="0.4" fill="currentColor" opacity="0.75"/>
  <rect x="9.5" y="5"  width="2.5" height="8" rx="0.4" fill="currentColor" opacity="0.55"/>
  <line x1="1" y1="13.5" x2="13" y2="13.5" stroke="currentColor" stroke-width="0.8"/>
</svg>`;

// ---------------------------------------------------------------------------
// PivotalExplorerWidget — left sidebar object inspector
// ---------------------------------------------------------------------------

export class PivotalExplorerWidget extends Widget {
  private _items: ExplorerItem[] = [];
  private _expanded: Set<string> = new Set();
  private _clickCb: ((name: string) => void) | null = null;
  private _listEl!: HTMLElement;

  constructor() {
    super();
    this.addClass('pv-explorer');
    this.id = 'pivotal-explorer-panel';
    this.title.label = 'Pivotal';
    this.title.caption = 'Pivotal Object Explorer';
    this.title.closable = false;

    this.node.innerHTML = `
      <div class="pv-explorer-header">Pivotal Objects</div>
      <div class="pv-explorer-list"></div>
    `;

    this._listEl = this.node.querySelector('.pv-explorer-list') as HTMLElement;
    this._renderEmpty();
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  setItemClickCallback(cb: (name: string) => void): void {
    this._clickCb = cb;
  }

  setItems(items: ExplorerItem[]): void {
    this._items = items;
    // Remove expanded entries that no longer exist
    for (const name of this._expanded) {
      if (!items.find(it => it.name === name)) this._expanded.delete(name);
    }
    this._render();
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  private _renderEmpty(): void {
    this._listEl.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'pv-explorer-empty';
    empty.textContent = 'No objects yet';
    this._listEl.appendChild(empty);
  }

  private _render(): void {
    this._listEl.innerHTML = '';
    if (!this._items.length) {
      this._renderEmpty();
      return;
    }
    for (const item of this._items) {
      this._renderItem(item);
    }
  }

  private _renderItem(item: ExplorerItem): void {
    const isExpanded = this._expanded.has(item.name);
    const hasColumns = item.type === 'dataframe' && !!(item.columns?.length);

    // --- Row ---
    const row = document.createElement('div');
    row.className = 'pv-explorer-row';
    row.setAttribute('role', 'row');

    const toggle = document.createElement('span');
    toggle.className = 'pv-explorer-toggle';
    toggle.textContent = hasColumns ? (isExpanded ? '▼' : '▶') : '';
    toggle.setAttribute('aria-hidden', 'true');

    const icon = document.createElement('span');
    icon.className = 'pv-explorer-icon';
    icon.innerHTML = item.type === 'dataframe' ? DF_ICON : CHART_ICON;

    const nameEl = document.createElement('span');
    nameEl.className = 'pv-explorer-name';
    nameEl.textContent = item.name;
    nameEl.title = item.name;

    row.appendChild(toggle);
    row.appendChild(icon);
    row.appendChild(nameEl);

    if (item.shape) {
      const shapeEl = document.createElement('span');
      shapeEl.className = 'pv-explorer-shape';
      shapeEl.textContent = `${item.shape[0].toLocaleString()}×${item.shape[1]}`;
      row.appendChild(shapeEl);
    }

    // Toggle expand on clicking the toggle arrow
    toggle.addEventListener('click', e => {
      e.stopPropagation();
      if (!hasColumns) return;
      if (isExpanded) this._expanded.delete(item.name);
      else this._expanded.add(item.name);
      this._render();
    });

    // Focus viewer on clicking the row (except the toggle)
    row.addEventListener('click', () => {
      this._clickCb?.(item.name);
    });

    this._listEl.appendChild(row);

    // --- Column tree (when expanded) ---
    if (hasColumns && isExpanded) {
      const colList = document.createElement('div');
      colList.className = 'pv-explorer-cols';
      for (const col of item.columns!) {
        const colRow = document.createElement('div');
        colRow.className = 'pv-explorer-col';

        const colName = document.createElement('span');
        colName.className = 'pv-explorer-col-name';
        colName.textContent = col.name;
        colName.title = col.name;

        const colDtype = document.createElement('span');
        colDtype.className = 'pv-explorer-col-dtype';
        colDtype.textContent = col.dtype;

        colRow.appendChild(colName);
        colRow.appendChild(colDtype);
        colList.appendChild(colRow);
      }
      this._listEl.appendChild(colList);
    }
  }
}
