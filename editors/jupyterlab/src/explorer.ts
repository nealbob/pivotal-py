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

const GT_TABLE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <rect x="1" y="1.5" width="12" height="2.5" rx="0.4" fill="currentColor" opacity="0.85"/>
  <line x1="1" y1="6"   x2="13" y2="6"   stroke="currentColor" stroke-width="0.8" opacity="0.55"/>
  <line x1="1" y1="8.5" x2="13" y2="8.5" stroke="currentColor" stroke-width="0.8" opacity="0.45"/>
  <line x1="1" y1="11"  x2="13" y2="11"  stroke="currentColor" stroke-width="0.8" opacity="0.35"/>
  <line x1="1" y1="13"  x2="13" y2="13"  stroke="currentColor" stroke-width="1.2" opacity="0.6"/>
</svg>`;

const EYE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12" width="12" height="12">
  <ellipse cx="6" cy="6" rx="5" ry="3.5" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <circle cx="6" cy="6" r="1.8" fill="currentColor" opacity="0.85"/>
</svg>`;

// ---------------------------------------------------------------------------
// PivotalExplorerWidget — left sidebar object inspector
// ---------------------------------------------------------------------------

export class PivotalExplorerWidget extends Widget {
  private _items: ExplorerItem[] = [];
  private _expanded: Set<string> = new Set();
  private _clickCb: ((name: string) => void) | null = null;
  private _deleteCb: ((name: string) => void) | null = null;
  private _currentTable: string | null = null;
  private _viewingItem: string | null = null;
  private _focusedName: string | null = null;
  private _listEl!: HTMLElement;
  private _statusEl!: HTMLElement;
  private _contextMenu: HTMLElement | null = null;

  constructor() {
    super();
    this.addClass('pv-explorer');
    this.id = 'pivotal-explorer-panel';
    this.title.label = '';
    this.title.caption = 'Pivotal Object Explorer';
    this.title.closable = false;

    this.node.innerHTML = `
      <div class="pv-explorer-header">Pivotal Objects</div>
      <div class="pv-explorer-list"></div>
      <div class="pv-explorer-status">No active table</div>
    `;

    this._listEl   = this.node.querySelector('.pv-explorer-list')   as HTMLElement;
    this._statusEl = this.node.querySelector('.pv-explorer-status') as HTMLElement;
    this._renderEmpty();

    // Del key deletes the focused item
    this.node.tabIndex = -1;
    this.node.addEventListener('keydown', e => {
      if (e.key === 'Delete' && this._focusedName) {
        this._deleteCb?.(this._focusedName);
      }
    });

    // Dismiss context menu on outside click
    document.addEventListener('click', () => this._dismissContextMenu(), true);
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  setItemClickCallback(cb: (name: string) => void): void {
    this._clickCb = cb;
  }

  setDeleteCallback(cb: (name: string) => void): void {
    this._deleteCb = cb;
  }

  setItems(items: ExplorerItem[]): void {
    this._items = items;
    for (const name of this._expanded) {
      if (!items.find(it => it.name === name)) this._expanded.delete(name);
    }
    // If the current table was deleted, clear the status bar
    if (this._currentTable && !items.find(it => it.name === this._currentTable)) {
      this._currentTable = null;
      this._statusEl.textContent = 'No active table';
    }
    this._render();
  }

  setCurrentTable(name: string | null): void {
    this._currentTable = name;
    this._render();
    this._statusEl.textContent = name ? `Current df: ${name}` : 'No active table';
  }

  setViewingItem(name: string | null): void {
    this._viewingItem = name;
    this._render();
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  private _dismissContextMenu(): void {
    if (this._contextMenu) {
      this._contextMenu.remove();
      this._contextMenu = null;
    }
  }

  private _showContextMenu(x: number, y: number, name: string): void {
    this._dismissContextMenu();
    const menu = document.createElement('div');
    menu.className = 'pv-explorer-context-menu';
    const deleteItem = document.createElement('div');
    deleteItem.className = 'pv-explorer-context-item';
    deleteItem.textContent = 'Delete';
    deleteItem.addEventListener('click', e => {
      e.stopPropagation();
      this._dismissContextMenu();
      this._deleteCb?.(name);
    });
    menu.appendChild(deleteItem);
    menu.style.left = `${x}px`;
    menu.style.top  = `${y}px`;
    document.body.appendChild(menu);
    this._contextMenu = menu;
  }

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
    const isCurrent  = item.type === 'dataframe' && item.name === this._currentTable;
    const isViewing  = item.name === this._viewingItem;

    // --- Row ---
    const row = document.createElement('div');
    row.className = 'pv-explorer-row';
    if (isCurrent) row.classList.add('pv-current-table');
    row.setAttribute('role', 'row');

    const toggle = document.createElement('span');
    toggle.className = 'pv-explorer-toggle';
    toggle.textContent = hasColumns ? (isExpanded ? '▼' : '▶') : '';
    toggle.setAttribute('aria-hidden', 'true');

    const icon = document.createElement('span');
    icon.className = 'pv-explorer-icon';
    icon.innerHTML = item.type === 'dataframe' ? DF_ICON
                   : item.type === 'chart'     ? CHART_ICON
                   : GT_TABLE_ICON;

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

    if (isViewing) {
      const eyeEl = document.createElement('span');
      eyeEl.className = 'pv-explorer-eye';
      eyeEl.innerHTML = EYE_ICON;
      eyeEl.title = 'Currently shown in viewer';
      row.appendChild(eyeEl);
    }

    // Toggle expand on clicking the toggle arrow
    toggle.addEventListener('click', e => {
      e.stopPropagation();
      if (!hasColumns) return;
      if (isExpanded) this._expanded.delete(item.name);
      else this._expanded.add(item.name);
      this._render();
    });

    // Track focused item and focus viewer on click
    row.addEventListener('click', () => {
      this._focusedName = item.name;
      this.node.focus();
      this._clickCb?.(item.name);
    });

    // Right-click context menu
    row.addEventListener('contextmenu', e => {
      e.preventDefault();
      e.stopPropagation();
      this._focusedName = item.name;
      this._showContextMenu(e.clientX, e.clientY, item.name);
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
