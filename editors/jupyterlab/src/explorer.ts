import { Widget } from '@lumino/widgets';
import { ExplorerItem, ValueInfo } from './viewer';

const DF_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <rect x="1" y="1" width="12" height="3" rx="0.5" fill="currentColor" opacity="0.75"/>
  <rect x="1" y="5.5" width="5" height="2.2" rx="0.4" fill="currentColor" opacity="0.55"/>
  <rect x="8" y="5.5" width="5" height="2.2" rx="0.4" fill="currentColor" opacity="0.55"/>
  <rect x="1" y="9.3" width="5" height="2.2" rx="0.4" fill="currentColor" opacity="0.4"/>
  <rect x="8" y="9.3" width="5" height="2.2" rx="0.4" fill="currentColor" opacity="0.4"/>
</svg>`;

const CHART_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <rect x="1" y="7" width="2.5" height="6" rx="0.4" fill="currentColor" opacity="0.6"/>
  <rect x="5" y="3" width="2.5" height="10" rx="0.4" fill="currentColor" opacity="0.75"/>
  <rect x="9.5" y="5" width="2.5" height="8" rx="0.4" fill="currentColor" opacity="0.55"/>
  <line x1="1" y1="13.5" x2="13" y2="13.5" stroke="currentColor" stroke-width="0.8"/>
</svg>`;

const GT_TABLE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <rect x="1" y="1.5" width="12" height="2.5" rx="0.4" fill="currentColor" opacity="0.85"/>
  <line x1="1" y1="6" x2="13" y2="6" stroke="currentColor" stroke-width="0.8" opacity="0.55"/>
  <line x1="1" y1="8.5" x2="13" y2="8.5" stroke="currentColor" stroke-width="0.8" opacity="0.45"/>
  <line x1="1" y1="11" x2="13" y2="11" stroke="currentColor" stroke-width="0.8" opacity="0.35"/>
  <line x1="1" y1="13" x2="13" y2="13" stroke="currentColor" stroke-width="1.2" opacity="0.6"/>
</svg>`;

const SCALAR_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <circle cx="7" cy="7" r="3.2" fill="currentColor" opacity="0.82"/>
  <circle cx="7" cy="7" r="1.2" fill="currentColor" opacity="0.95"/>
</svg>`;

const LIST_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <circle cx="3" cy="3.2" r="1" fill="currentColor" opacity="0.8"/>
  <circle cx="3" cy="7" r="1" fill="currentColor" opacity="0.7"/>
  <circle cx="3" cy="10.8" r="1" fill="currentColor" opacity="0.6"/>
  <line x1="5.3" y1="3.2" x2="11.5" y2="3.2" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" opacity="0.85"/>
  <line x1="5.3" y1="7" x2="11.5" y2="7" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" opacity="0.75"/>
  <line x1="5.3" y1="10.8" x2="11.5" y2="10.8" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" opacity="0.65"/>
</svg>`;

const DICT_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <rect x="2" y="2" width="3.1" height="3.1" rx="0.6" fill="currentColor" opacity="0.8"/>
  <rect x="8.9" y="2" width="3.1" height="3.1" rx="0.6" fill="currentColor" opacity="0.7"/>
  <rect x="5.45" y="8.9" width="3.1" height="3.1" rx="0.6" fill="currentColor" opacity="0.6"/>
  <path d="M5.1 3.55h3.8M7 5.1v2.65" stroke="currentColor" stroke-width="1" stroke-linecap="round" opacity="0.78"/>
</svg>`;

const EYE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12" width="12" height="12">
  <ellipse cx="6" cy="6" rx="5" ry="3.5" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <circle cx="6" cy="6" r="1.8" fill="currentColor" opacity="0.85"/>
</svg>`;

type ExplorerGroup = [ExplorerItem[], string, string];

export class PivotalExplorerWidget extends Widget {
  private _items: ExplorerItem[] = [];
  private _expanded: Set<string> = new Set();
  private _collapsedFolders: Set<string> = new Set();
  private _clickCb: ((name: string) => void) | null = null;
  private _colClickCb: ((tableName: string, colName: string) => void) | null = null;
  private _deleteCb: ((name: string) => void) | null = null;
  private _currentTable: string | null = null;
  private _viewingItem: string | null = null;
  private _focusedName: string | null = null;
  private _listEl!: HTMLElement;
  private _currentTableChangedCb: ((name: string | null) => void) | null = null;
  private _contextMenu: HTMLElement | null = null;

  constructor() {
    super();
    this.addClass('pv-explorer');
    this.id = 'pivotal-explorer-panel';
    this.title.label = '';
    this.title.caption = 'Pivotal Object Explorer';
    this.title.closable = false;

    this.node.innerHTML = `
      <div class="pv-explorer-header">
        <span class="pv-explorer-title">Pivotal Objects</span>
      </div>
      <div class="pv-explorer-list"></div>
    `;

    this._listEl = this.node.querySelector('.pv-explorer-list') as HTMLElement;
    this._renderEmpty();

    this.node.tabIndex = -1;
    let lastKey = '';
    let lastKeyTime = 0;
    this.node.addEventListener('keydown', e => {
      const nav = this._buildNavList();
      const idx = this._focusedName ? nav.indexOf(this._focusedName) : -1;

      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        if (!nav.length) return;
        this._setFocus(idx < nav.length - 1 ? nav[idx + 1] : nav[0]);
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (!nav.length) return;
        this._setFocus(idx > 0 ? nav[idx - 1] : nav[nav.length - 1]);
      } else if ((e.key === 'l' || e.key === 'ArrowRight') && this._focusedName) {
        this._activateFocused(true);
      } else if ((e.key === 'h' || e.key === 'ArrowLeft') && this._focusedName) {
        this._collapseFocused();
      } else if ((e.key === 'Enter' || e.key === ' ') && this._focusedName) {
        this._activateFocused(false);
      } else if (e.key === 'Delete' && this._focusedName) {
        this._deleteCb?.(this._getRootName(this._focusedName));
      } else if (e.key === 'd' && this._focusedName) {
        const now = Date.now();
        if (lastKey === 'd' && now - lastKeyTime < 500) {
          this._deleteCb?.(this._getRootName(this._focusedName));
        }
        lastKey = 'd';
        lastKeyTime = now;
        return;
      }

      lastKey = e.key;
      lastKeyTime = Date.now();
    });

    document.addEventListener('click', () => {
      this._dismissContextMenu();
    }, true);
  }

  setItemClickCallback(cb: (name: string) => void): void {
    this._clickCb = cb;
  }

  setColClickCallback(cb: (tableName: string, colName: string) => void): void {
    this._colClickCb = cb;
  }

  setDeleteCallback(cb: (name: string) => void): void {
    this._deleteCb = cb;
  }

  getCurrentTable(): string | null {
    return this._currentTable;
  }

  setItems(items: ExplorerItem[]): void {
    this._items = items;
    for (const key of [...this._expanded]) {
      if (!this._keyExists(key)) this._expanded.delete(key);
    }
    if (this._currentTable && !items.find(it => it.name === this._currentTable)) {
      this._currentTable = null;
      this._currentTableChangedCb?.(null);
    }
    if (this._focusedName && !this._keyExists(this._focusedName)) {
      this._focusedName = null;
    }
    this._render();
  }

  setCurrentTable(name: string | null): void {
    this._currentTable = name;
    this._render();
    this._currentTableChangedCb?.(name);
  }

  setCurrentTableChangedCallback(cb: (name: string | null) => void): void {
    this._currentTableChangedCb = cb;
  }

  setViewingItem(name: string | null): void {
    this._viewingItem = name;
    this._render();
  }

  private _groups(): ExplorerGroup[] {
    return [
      [this._items.filter(it => it.type === 'dataframe'), 'data', 'Data'],
      [this._items.filter(it => it.type === 'chart'), 'charts', 'Charts'],
      [this._items.filter(it => it.type === 'gt_table'), 'tables', 'Tables'],
      [this._items.filter(it => it.type === 'value'), 'values', 'Values'],
    ];
  }

  private _getRootName(key: string): string {
    return key.split('::')[0];
  }

  private _getValueInfoByPath(path: string[]): ValueInfo | undefined {
    if (!path.length) return undefined;
    const root = this._items.find(it => it.type === 'value' && it.name === path[0])?.value;
    let current = root;
    for (let i = 1; i < path.length && current; i++) {
      current = current.children?.[path[i]];
    }
    return current;
  }

  private _isValueExpandable(value: ValueInfo | undefined): boolean {
    return !!value?.children && Object.keys(value.children).length > 0;
  }

  private _hasChildren(item: ExplorerItem): boolean {
    if (item.type === 'dataframe') return !!item.columns?.length;
    if (item.type === 'value') return this._isValueExpandable(item.value);
    return false;
  }

  private _keyExists(key: string): boolean {
    if (!key.includes('::')) return this._items.some(it => it.name === key);
    const parts = key.split('::');
    return !!this._getValueInfoByPath(parts);
  }

  private _valueSummary(value: ValueInfo | undefined): string {
    if (!value) return '';
    if (value.kind === 'scalar') {
      const preview = value.preview ?? '';
      const suffix = value.value_type ? ` <${value.value_type}>` : '';
      return `${preview}${suffix}`.trim();
    }
    if (value.kind === 'list') {
      const length = value.length ?? 0;
      return `${length} item${length === 1 ? '' : 's'}`;
    }
    const size = value.size ?? 0;
    return `${size} entr${size === 1 ? 'y' : 'ies'}`;
  }

  private _valueBadge(value: ValueInfo | undefined): string {
    if (!value || value.kind === 'scalar') return '';
    return value.kind;
  }

  private _valueIcon(value: ValueInfo | undefined): string {
    if (!value) return SCALAR_ICON;
    if (value.kind === 'list') return LIST_ICON;
    if (value.kind === 'dict') return DICT_ICON;
    return SCALAR_ICON;
  }

  private _buildNavList(): string[] {
    const list: string[] = [];
    for (const [items, folderId] of this._groups()) {
      if (!items.length || this._collapsedFolders.has(folderId)) continue;
      for (const item of items) {
        list.push(item.name);
        if (item.type === 'dataframe' && this._expanded.has(item.name) && item.columns?.length) {
          for (const col of item.columns) {
            list.push(`${item.name}::${col.name}`);
          }
        }
        if (item.type === 'value' && this._expanded.has(item.name)) {
          this._appendValueNav(list, item.name, item.value);
        }
      }
    }
    return list;
  }

  private _appendValueNav(list: string[], prefix: string, value: ValueInfo | undefined): void {
    if (!value?.children) return;
    for (const [childKey, childValue] of Object.entries(value.children)) {
      const navKey = `${prefix}::${childKey}`;
      list.push(navKey);
      if (this._expanded.has(navKey)) {
        this._appendValueNav(list, navKey, childValue);
      }
    }
  }

  private _setFocus(key: string): void {
    this._focusedName = key;
    this._render();
    this._listEl.querySelector('.pv-focused')?.scrollIntoView({ block: 'nearest' });
  }

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
    menu.style.top = `${y}px`;
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

  private _activateFocused(expandOnly: boolean): void {
    if (!this._focusedName) return;
    if (this._focusedName.includes('::')) {
      const value = this._getValueInfoByPath(this._focusedName.split('::'));
      if (this._isValueExpandable(value) && !this._expanded.has(this._focusedName)) {
        this._expanded.add(this._focusedName);
        this._render();
      } else if (!expandOnly && this._isValueExpandable(value)) {
        this._expanded.delete(this._focusedName);
        this._render();
      }
      return;
    }

    const item = this._items.find(it => it.name === this._focusedName);
    if (!item) return;
    if (this._hasChildren(item) && !this._expanded.has(item.name)) {
      this._expanded.add(item.name);
      this._render();
      return;
    }
    if (!expandOnly) {
      if (item.type === 'value' && this._hasChildren(item)) {
        this._expanded.delete(item.name);
        this._render();
      } else if (item.type !== 'value') {
        this._clickCb?.(item.name);
      }
    } else if (item.type !== 'value') {
      this._clickCb?.(item.name);
    }
  }

  private _collapseFocused(): void {
    if (!this._focusedName) return;
    if (this._focusedName.includes('::')) {
      const parentKey = this._focusedName.split('::').slice(0, -1).join('::');
      this._expanded.delete(this._focusedName);
      this._focusedName = parentKey;
      this._render();
      return;
    }
    if (this._expanded.has(this._focusedName)) {
      this._expanded.delete(this._focusedName);
      this._render();
    }
  }

  private _render(): void {
    this._listEl.innerHTML = '';
    if (!this._items.length) {
      this._renderEmpty();
      return;
    }

    for (const [items, folderId, label] of this._groups()) {
      this._renderFolder(folderId, label, items);
    }
  }

  private _renderFolder(id: string, label: string, items: ExplorerItem[]): void {
    if (!items.length) return;

    const isCollapsed = this._collapsedFolders.has(id);
    const header = document.createElement('div');
    header.className = 'pv-explorer-folder-header';

    const toggle = document.createElement('span');
    toggle.className = 'pv-explorer-folder-toggle';
    toggle.textContent = isCollapsed ? '▶' : '▼';
    toggle.setAttribute('aria-hidden', 'true');

    const labelEl = document.createElement('span');
    labelEl.className = 'pv-explorer-folder-label';
    labelEl.textContent = label;

    const countEl = document.createElement('span');
    countEl.className = 'pv-explorer-folder-count';
    countEl.textContent = `${items.length}`;

    header.appendChild(toggle);
    header.appendChild(labelEl);
    header.appendChild(countEl);
    header.addEventListener('click', () => {
      if (isCollapsed) this._collapsedFolders.delete(id);
      else this._collapsedFolders.add(id);
      this._render();
    });

    this._listEl.appendChild(header);
    if (!isCollapsed) {
      for (const item of items) {
        this._renderItem(item);
      }
    }
  }

  private _renderItem(item: ExplorerItem): void {
    const isExpanded = this._expanded.has(item.name);
    const hasChildren = this._hasChildren(item);
    const isCurrent = item.type === 'dataframe' && item.name === this._currentTable;
    const isViewing = item.name === this._viewingItem;

    const row = document.createElement('div');
    row.className = 'pv-explorer-row';
    if (isCurrent) row.classList.add('pv-current-table');
    if (item.name === this._focusedName) row.classList.add('pv-focused');
    row.setAttribute('role', 'row');

    const toggle = document.createElement('span');
    toggle.className = 'pv-explorer-toggle';
    toggle.textContent = hasChildren ? (isExpanded ? '▼' : '▶') : '';
    toggle.setAttribute('aria-hidden', 'true');

    const icon = document.createElement('span');
    icon.className = 'pv-explorer-icon';
    icon.innerHTML = item.type === 'dataframe'
      ? DF_ICON
      : item.type === 'chart'
        ? CHART_ICON
        : item.type === 'gt_table'
          ? GT_TABLE_ICON
          : this._valueIcon(item.value);

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
    } else if (item.type === 'value') {
      const badge = this._valueBadge(item.value);
      if (badge) {
        const badgeEl = document.createElement('span');
        badgeEl.className = 'pv-explorer-col-type pv-col-type-string';
        badgeEl.textContent = badge;
        badgeEl.title = badge;
        row.appendChild(badgeEl);
      }
      const summaryEl = document.createElement('span');
      summaryEl.className = 'pv-explorer-shape';
      summaryEl.textContent = this._valueSummary(item.value);
      row.appendChild(summaryEl);
    }

    if (isViewing) {
      const eyeEl = document.createElement('span');
      eyeEl.className = 'pv-explorer-eye';
      eyeEl.innerHTML = EYE_ICON;
      eyeEl.title = 'Currently shown in viewer';
      row.appendChild(eyeEl);
    }

    toggle.addEventListener('click', e => {
      e.stopPropagation();
      if (!hasChildren) return;
      if (isExpanded) this._expanded.delete(item.name);
      else this._expanded.add(item.name);
      this._render();
    });

    row.addEventListener('click', () => {
      this._focusedName = item.name;
      this.node.focus();
      if (item.type !== 'value') this._clickCb?.(item.name);
      else if (hasChildren) {
        if (this._expanded.has(item.name)) this._expanded.delete(item.name);
        else this._expanded.add(item.name);
        this._render();
      } else {
        this._render();
      }
    });

    row.addEventListener('contextmenu', e => {
      e.preventDefault();
      e.stopPropagation();
      this._focusedName = item.name;
      this._showContextMenu(e.clientX, e.clientY, item.name);
    });

    this._listEl.appendChild(row);

    if (item.type === 'dataframe' && isExpanded) {
      const colList = document.createElement('div');
      colList.className = 'pv-explorer-cols';
      for (const col of item.columns ?? []) {
        const navKey = `${item.name}::${col.name}`;
        const colRow = document.createElement('div');
        colRow.className = 'pv-explorer-col';
        if (this._focusedName === navKey) colRow.classList.add('pv-focused');

        const colName = document.createElement('span');
        colName.className = 'pv-explorer-col-name';
        colName.textContent = col.name;
        colName.title = col.name;

        const colDtype = document.createElement('span');
        colDtype.className = 'pv-explorer-col-dtype';
        colDtype.textContent = col.dtype;

        const colType = document.createElement('span');
        colType.className = `pv-explorer-col-type pv-col-type-${col.col_type ?? 'string'}`;
        colType.textContent = col.col_type ?? '';
        colType.title = col.col_type ?? '';

        colRow.style.cursor = 'pointer';
        colRow.title = `Navigate to column "${col.name}"`;
        colRow.addEventListener('click', e => {
          e.stopPropagation();
          this._focusedName = navKey;
          this._colClickCb?.(item.name, col.name);
          this._render();
        });

        colRow.appendChild(colName);
        colRow.appendChild(colType);
        colRow.appendChild(colDtype);
        colList.appendChild(colRow);
      }
      this._listEl.appendChild(colList);
    }

    if (item.type === 'value' && isExpanded) {
      this._renderValueChildren(item.name, item.value, 1);
    }
  }

  private _renderValueChildren(prefix: string, value: ValueInfo | undefined, depth: number): void {
    if (!value?.children || !Object.keys(value.children).length) return;

    const childList = document.createElement('div');
    childList.className = 'pv-explorer-cols pv-explorer-values';
    childList.style.marginLeft = `${22 + Math.max(0, depth - 1) * 14}px`;

    const appendChildren = (container: HTMLElement, keyPrefix: string, info: ValueInfo, level: number) => {
      if (!info.children) return;
      for (const [childKey, childValue] of Object.entries(info.children)) {
        const navKey = `${keyPrefix}::${childKey}`;
        const childRow = document.createElement('div');
        childRow.className = 'pv-explorer-col';
        if (this._focusedName === navKey) childRow.classList.add('pv-focused');

        const toggle = document.createElement('span');
        toggle.className = 'pv-explorer-toggle';
        toggle.textContent = this._isValueExpandable(childValue)
          ? (this._expanded.has(navKey) ? '▼' : '▶')
          : '';

        const nameEl = document.createElement('span');
        nameEl.className = 'pv-explorer-col-name';
        nameEl.textContent = childKey;
        nameEl.title = childKey;

        const iconEl = document.createElement('span');
        iconEl.className = 'pv-explorer-icon pv-explorer-value-icon';
        iconEl.innerHTML = this._valueIcon(childValue);

        const summaryEl = document.createElement('span');
        summaryEl.className = 'pv-explorer-col-dtype';
        summaryEl.textContent = this._valueSummary(childValue);

        const badge = this._valueBadge(childValue);
        if (badge) {
          const kindEl = document.createElement('span');
          kindEl.className = 'pv-explorer-col-type pv-col-type-string';
          kindEl.textContent = badge;
          kindEl.title = badge;
          childRow.appendChild(toggle);
          childRow.appendChild(iconEl);
          childRow.appendChild(nameEl);
          childRow.appendChild(kindEl);
          childRow.appendChild(summaryEl);
        } else {
          childRow.appendChild(toggle);
          childRow.appendChild(iconEl);
          childRow.appendChild(nameEl);
          childRow.appendChild(summaryEl);
        }

        childRow.addEventListener('click', e => {
          e.stopPropagation();
          this._focusedName = navKey;
          if (this._isValueExpandable(childValue)) {
            if (this._expanded.has(navKey)) this._expanded.delete(navKey);
            else this._expanded.add(navKey);
          }
          this._render();
        });

        toggle.addEventListener('click', e => {
          e.stopPropagation();
          if (!this._isValueExpandable(childValue)) return;
          if (this._expanded.has(navKey)) this._expanded.delete(navKey);
          else this._expanded.add(navKey);
          this._render();
        });

        container.appendChild(childRow);

        if (this._expanded.has(navKey)) {
          const nested = document.createElement('div');
          nested.className = 'pv-explorer-cols pv-explorer-values';
          nested.style.marginLeft = `${22 + level * 14}px`;
          container.appendChild(nested);
          appendChildren(nested, navKey, childValue, level + 1);
        }
      }
    };

    appendChildren(childList, prefix, value, depth);
    this._listEl.appendChild(childList);
  }
}
