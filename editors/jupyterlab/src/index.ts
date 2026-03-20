import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
} from '@jupyterlab/application';

import { ISettingRegistry } from '@jupyterlab/settingregistry';

import { LabIcon } from '@jupyterlab/ui-components';

import {
  IEditorLanguageRegistry,
  IEditorExtensionRegistry,
} from '@jupyterlab/codemirror';
import { LanguageSupport } from '@codemirror/language';
import { Compartment, Prec, RangeSetBuilder } from '@codemirror/state';
import { Decoration, DecorationSet, EditorView, ViewPlugin, ViewUpdate } from '@codemirror/view';
import {
  autocompletion,
  CompletionContext,
  CompletionResult,
  Completion,
} from '@codemirror/autocomplete';
import { INotebookTracker, NotebookActions } from '@jupyterlab/notebook';
import { ToolbarButton, Notification } from '@jupyterlab/apputils';
import { IStatusBar } from '@jupyterlab/statusbar';
import { PageConfig } from '@jupyterlab/coreutils';
import { Menu, Widget } from '@lumino/widgets';

import { pivotalLanguage } from './language';
import { PivotalViewerWidget, ViewerMessage } from './viewer';
import { PivotalExplorerWidget } from './explorer';

const MAGIC_RE = /^%%pivotal(\s|$)/;

// ---------------------------------------------------------------------------
// Icon
// ---------------------------------------------------------------------------

const PIVOTAL_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
  <rect x="1" y="1" width="14" height="14" rx="2" fill="#616161"/>
  <rect x="2.5" y="2.5" width="11" height="3.5" rx="0.8" fill="#E8EAED" opacity="0.9"/>
  <rect x="2.5" y="7"  width="4.5" height="2.5" rx="0.5" fill="#E8EAED" opacity="0.6"/>
  <rect x="9"   y="7"  width="4.5" height="2.5" rx="0.5" fill="#E8EAED" opacity="0.6"/>
  <rect x="2.5" y="11" width="4.5" height="2.5" rx="0.5" fill="#E8EAED" opacity="0.4"/>
  <rect x="9"   y="11" width="4.5" height="2.5" rx="0.5" fill="#E8EAED" opacity="0.4"/>
</svg>`;

export const pivotalIcon = new LabIcon({
  name: 'pivotal:file',
  svgstr: PIVOTAL_SVG,
});

// Greyscale pivot-wheel logo for the left sidebar.
// Same geometry as pivotal_logo.svg but rendered with currentColor so it
// adapts to both JupyterLab light and dark themes. No background rect so
// the sidebar background shows through.
const PIVOTAL_GREY_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
  <!-- Dim track ring -->
  <circle cx="64" cy="64" r="40" fill="none"
          stroke="currentColor" stroke-width="15" opacity="0.15"/>
  <!-- Arc 1: top-right (brightest) -->
  <path d="M 67.49 24.15 A 40 40 0 0 1 100.25 80.90"
        fill="none" stroke="currentColor" stroke-width="15"
        stroke-linecap="round" opacity="0.9"/>
  <!-- Arc 2: bottom (mid) -->
  <path d="M 96.77 86.94 A 40 40 0 0 1 31.23 86.94"
        fill="none" stroke="currentColor" stroke-width="15"
        stroke-linecap="round" opacity="0.6"/>
  <!-- Arc 3: left (softest) -->
  <path d="M 27.75 80.90 A 40 40 0 0 1 60.51 24.15"
        fill="none" stroke="currentColor" stroke-width="15"
        stroke-linecap="round" opacity="0.4"/>
  <!-- Radial spoke lines -->
  <line x1="64" y1="64" x2="108.17" y2="47.93"
        stroke="currentColor" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
  <line x1="64" y1="64" x2="40.50" y2="23.30"
        stroke="currentColor" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
  <!-- Centre pivot dot -->
  <circle cx="64" cy="64" r="8.5" fill="currentColor" opacity="0.9"/>
</svg>`;

export const pivotalGreyIcon = new LabIcon({
  name: 'pivotal:explorer',
  svgstr: PIVOTAL_GREY_SVG,
});

// ---------------------------------------------------------------------------
// Autocomplete types
// ---------------------------------------------------------------------------

interface TableInfo {
  columns: (string | string[])[];
  shape: [number, number];
  dtypes: Record<string, string>;
}

interface AutocompleteData {
  tables: Record<string, TableInfo>;
  current_table?: string;
}

type CompletionCtx =
  | { type: 'command' }
  | { type: 'table' }
  | { type: 'column'; table: string }
  | { type: 'agg' }
  | { type: 'charttype' }
  | { type: 'none' };

const COMMAND_KEYWORDS = [
  'df', 'load', 'filter', 'select', 'sort', 'group by',
  'merge', 'left merge', 'right merge', 'inner merge', 'outer merge',
  'concat', 'pivot', 'unpivot', 'plot', 'agg plot', 'drop', 'rename', 'fillna', 'dropna',
  'distinct', 'python', 'save', 'apply', 'table', 'delete',
  'rank', 'lag', 'lead', 'cumsum', 'cummean', 'cummin', 'cummax', 'rolling',
];

const AGG_KEYWORDS = ['mean', 'sum', 'count', 'min', 'max', 'median', 'std', 'avg'];
const CHART_TYPES = ['line', 'bar', 'scatter', 'hist', 'box', 'area'];

// ---------------------------------------------------------------------------
// Autocomplete file fetching — last_modified-cached
// ---------------------------------------------------------------------------

let _acCache: { path: string; lastModified: string; data: AutocompleteData } | null = null;

// Set of simple-identifier column names across all known tables, for highlighting.
let _colNameSet: Set<string> = new Set();

function _updateColNameSet(ac: AutocompleteData): void {
  const next = new Set<string>();
  for (const tableInfo of Object.values(ac.tables)) {
    for (const col of tableInfo.columns) {
      const name = Array.isArray(col) ? col[col.length - 1] : String(col);
      if (/^[A-Za-z_]\w*$/.test(name)) next.add(name);
    }
  }
  _colNameSet = next;
}

async function fetchAutocompleteData(dir: string): Promise<AutocompleteData | null> {
  const acPath = dir ? `${dir}/pivotal_autocomplete.json` : 'pivotal_autocomplete.json';
  try {
    const resp = await fetch(`/api/contents/${acPath}`, { cache: 'no-store' });
    if (!resp.ok) return null;
    const json = await resp.json() as { content: string; last_modified: string };
    const lastModified = json.last_modified ?? '';
    if (_acCache && _acCache.path === acPath && _acCache.lastModified === lastModified) {
      return _acCache.data;
    }
    const data = JSON.parse(json.content) as AutocompleteData;
    _acCache = { path: acPath, lastModified, data };
    _updateColNameSet(data);
    return data;
  } catch {
    return null;
  }
}

// ViewPlugin that decorates column names with italic + underline.
// Reads from _colNameSet which is updated whenever autocomplete data is fetched.
const colNameMark = Decoration.mark({ class: 'pv-col-name' });

const colHighlightPlugin = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;
    constructor(view: EditorView) { this.decorations = this._build(view); }
    update(u: ViewUpdate) {
      if (u.docChanged || u.viewportChanged) this.decorations = this._build(u.view);
    }
    _build(view: EditorView): DecorationSet {
      const builder = new RangeSetBuilder<Decoration>();
      if (!_colNameSet.size) return builder.finish();
      const wordRe = /\b([A-Za-z_]\w*)\b/g;
      for (const { from, to } of view.visibleRanges) {
        const text = view.state.sliceDoc(from, to);
        let m: RegExpExecArray | null;
        wordRe.lastIndex = 0;
        while ((m = wordRe.exec(text)) !== null) {
          if (_colNameSet.has(m[1])) {
            builder.add(from + m.index, from + m.index + m[1].length, colNameMark);
          }
        }
      }
      return builder.finish();
    }
  },
  { decorations: v => v.decorations }
);

function getNotebookDir(app: JupyterFrontEnd): string {
  const widget = app.shell.currentWidget;
  if (widget && 'context' in widget) {
    const ctxPath = (widget as any).context?.path as string | undefined;
    if (ctxPath) {
      return ctxPath.includes('/')
        ? ctxPath.slice(0, ctxPath.lastIndexOf('/'))
        : '';
    }
  }
  return '';
}

// ---------------------------------------------------------------------------
// Context detection
// ---------------------------------------------------------------------------

function findActiveTable(
  lines: string[],
  cursorLine: number,
  ac: AutocompleteData | null,
): string | null {
  for (let i = cursorLine; i >= 0; i--) {
    const t = lines[i].trimStart();
    const dfM = t.match(/^df\s+(\w+)/);
    if (dfM) return dfM[1];
    const loadM = t.match(/^load\s+(\w+)/);
    if (loadM) return loadM[1];
  }
  return ac?.current_table ?? null;
}

function detectContext(
  lines: string[],
  cursorLine: number,
  cursorCol: number,
  ac: AutocompleteData | null,
): CompletionCtx {
  const raw = lines[cursorLine] ?? '';
  const upToCursor = raw.substring(0, cursorCol);
  const trimmed = upToCursor.trimStart();
  const indent = upToCursor.length - trimmed.length;

  if (trimmed === '' && indent === 0) return { type: 'command' };

  if (/^df\s+\w*$/.test(trimmed)) return { type: 'table' };
  if (/^df\s+\w+\s+from\s+\w*$/.test(trimmed)) return { type: 'table' };
  if (/^(left\s+|right\s+|inner\s+|outer\s+)?(merge|concat)\s+\w*$/.test(trimmed)) {
    return { type: 'table' };
  }

  if (/^plot\s+\w*$/.test(trimmed)) return { type: 'charttype' };

  if (/^agg\s+\w*$/.test(trimmed)) return { type: 'agg' };
  if (/^agg\s+\w+\s+\w*$/.test(trimmed)) {
    const table = findActiveTable(lines, cursorLine, ac);
    return table ? { type: 'column', table } : { type: 'none' };
  }

  const table = findActiveTable(lines, cursorLine, ac);
  if (table) {
    if (/^(filter|select|drop|distinct|sort|rename)\b/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^\w+\s*=/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^col\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^where\b/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^(group\s+by|by)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^(x|y|by|c)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
  }

  return { type: 'command' };
}

// ---------------------------------------------------------------------------
// Build completion items
// ---------------------------------------------------------------------------

function buildCompletions(ctx: CompletionCtx, ac: AutocompleteData | null): Completion[] {
  switch (ctx.type) {
    case 'command':
      return COMMAND_KEYWORDS.map(kw => ({ label: kw, type: 'keyword' }));
    case 'table':
      if (!ac) return [];
      return Object.keys(ac.tables).map(t => ({ label: t, type: 'variable' }));
    case 'column': {
      if (!ac) return [];
      const info = ac.tables[ctx.table];
      if (!info) return [];
      return info.columns.map(col => {
        const label = Array.isArray(col) ? col.join('.') : String(col);
        const dtype = info.dtypes?.[label];
        return { label, type: 'property', detail: dtype };
      });
    }
    case 'agg':
      return AGG_KEYWORDS.map(kw => ({ label: kw, type: 'function' }));
    case 'charttype':
      return CHART_TYPES.map(ct => ({ label: ct, type: 'enum' }));
    case 'none':
      return [];
  }
}

// ---------------------------------------------------------------------------
// CompletionSource
// ---------------------------------------------------------------------------

function makePivotalCompletionSource(app: JupyterFrontEnd) {
  return async function pivotalCompletionSource(
    context: CompletionContext
  ): Promise<CompletionResult | null> {
    const state = context.state;

    const lines: string[] = [];
    for (let i = 1; i <= state.doc.lines; i++) {
      lines.push(state.doc.line(i).text);
    }

    // Skip the %%pivotal magic line in notebook cells
    const lineOffset = MAGIC_RE.test(lines[0] ?? '') ? 1 : 0;

    const lineInfo = state.doc.lineAt(context.pos);
    const cursorLine = lineInfo.number - 1 - lineOffset;
    const cursorCol = context.pos - lineInfo.from;

    if (cursorLine < 0) return null;

    const effectiveLines = lines.slice(lineOffset);
    const dir = getNotebookDir(app);
    const ac = await fetchAutocompleteData(dir);

    const ctx = detectContext(effectiveLines, cursorLine, cursorCol, ac);
    const options = buildCompletions(ctx, ac);

    if (!options.length && !context.explicit) return null;

    const word = context.matchBefore(/\w*/);
    const from = word ? word.from : context.pos;

    return { from, options, validFor: /^\w*$/ };
  };
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@pivotal/jupyterlab:language',
  description: 'Syntax highlighting and autocomplete for the Pivotal data transformation DSL',
  autoStart: true,
  requires: [IEditorLanguageRegistry, IEditorExtensionRegistry],
  activate: (
    app: JupyterFrontEnd,
    languages: IEditorLanguageRegistry,
    extensions: IEditorExtensionRegistry,
  ) => {
    const completionSource = makePivotalCompletionSource(app);
    const completionExt = autocompletion({ override: [completionSource] });

    app.docRegistry.addFileType({
      name: 'pivotal',
      displayName: 'Pivotal',
      extensions: ['.pivotal'],
      mimeTypes: ['text/x-pivotal'],
      icon: pivotalIcon,
      contentType: 'file',
      fileFormat: 'text',
    });

    languages.addLanguage({
      name: 'pivotal',
      mime: 'text/x-pivotal',
      extensions: ['.pivotal'],
      load: async () => new LanguageSupport(pivotalLanguage, completionExt),
    });

    extensions.addExtension({
      name: '@pivotal/jupyterlab:magic-highlight',
      factory: () => {
        const compartment = new Compartment();
        let active = false;

        const ext = [
          compartment.of([]),
          EditorView.updateListener.of(update => {
            const firstLine = update.state.doc.line(1).text;
            const isPivotal = MAGIC_RE.test(firstLine);

            if (isPivotal === active) return;
            active = isPivotal;

            update.view.dispatch({
              effects: compartment.reconfigure(
                isPivotal
                  ? [
                      Prec.highest(new LanguageSupport(pivotalLanguage, completionExt)),
                      colHighlightPlugin,
                      EditorView.editorAttributes.of({ class: 'pv-pivotal-cell' }),
                    ]
                  : []
              ),
            });
          }),
        ];

        return { instance: () => ext, reconfigure: () => null };
      },
    });
  },
};

// ---------------------------------------------------------------------------
// Viewer plugin
// ---------------------------------------------------------------------------

let _viewerWidget: PivotalViewerWidget | null = null;
let _explorerWidget: PivotalExplorerWidget | null = null;

function getViewer(): PivotalViewerWidget {
  if (!_viewerWidget) {
    _viewerWidget = new PivotalViewerWidget();
  }
  return _viewerWidget;
}

function getExplorer(): PivotalExplorerWidget {
  if (!_explorerWidget) {
    _explorerWidget = new PivotalExplorerWidget();
  }
  return _explorerWidget;
}

const viewerPlugin: JupyterFrontEndPlugin<void> = {
  id: '@pivotal/jupyterlab:viewer',
  description: 'Object Viewer panel for DataFrames and charts from Pivotal cells',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ISettingRegistry, IStatusBar],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker, settings: ISettingRegistry | null, statusBar: IStatusBar | null) => {
    const viewer = getViewer();
    const explorer = getExplorer();
    explorer.title.icon = pivotalGreyIcon;
    viewer.title.icon = pivotalGreyIcon;

    app.shell.add(explorer, 'left', { rank: 100 });

    let viewerArea = 'right';

    // For 'right': add once at startup (sidebar widgets persist).
    // For 'main': add lazily on first use so the layout restorer doesn't
    //             clobber the tab before any data has arrived.
    const addViewerToRight = () => {
      app.shell.add(viewer, 'right', { rank: 900 });
    };

    const applyArea = (area: string) => {
      const newArea = (['right', 'main', 'down'] as string[]).includes(area) ? area : 'right';
      if (newArea === viewerArea && viewer.isAttached) return;
      // Detach from current area before moving
      if (viewer.isAttached) {
        viewer.parent = null;
      }
      viewerArea = newArea;
      if (viewerArea === 'right') {
        viewer.title.label = '';
        viewer.title.closable = false;
        addViewerToRight();
      }
      // 'main' and 'down' are added lazily on next activateViewer() call
    };

    if (settings) {
      settings.load('@pivotal/jupyterlab:plugin').then(s => {
        applyArea((s.get('viewerArea').composite as string) ?? 'right');
        s.changed.connect(() => {
          applyArea((s.get('viewerArea').composite as string) ?? 'right');
        });
      }).catch(() => {
        viewerArea = 'right';
        addViewerToRight();
      });
    } else {
      viewerArea = 'right';
      addViewerToRight();
    }

    const activateViewer = () => {
      if (viewerArea === 'main' || viewerArea === 'down') {
        // Add lazily if not currently attached (first call, or after tab was closed).
        // Lazy add avoids the layout restorer clobbering the panel before data arrives.
        if (!viewer.isAttached) {
          viewer.title.label = 'Pivotal Viewer';
          viewer.title.closable = true;
          const addOpts: Record<string, unknown> = { rank: 900 };
          if (viewerArea === 'main') {
            // Split the viewer to the right of the current notebook automatically.
            const refId = tracker.currentWidget?.id;
            if (refId) { addOpts['mode'] = 'split-right'; addOpts['ref'] = refId; }
          }
          app.shell.add(viewer, viewerArea as 'main' | 'down', addOpts as any);
        }
      }
      app.shell.activateById(viewer.id);
    };

    // Keep explorer in sync with viewer content and viewing state
    viewer.setContentChangedCallback(items => explorer.setItems(items));
    viewer.setViewingItemCallback(name => explorer.setViewingItem(name));

    // Insert a gui cell into the active notebook
    // Helper: insert a new code cell below the active cell and run it
    const insertAndRun = (code: string) => {
      const panel = tracker.currentWidget;
      if (!panel) return;
      const notebook = panel.content;
      NotebookActions.insertBelow(notebook);
      const active = notebook.activeCell;
      if (active) {
        active.model.sharedModel.setSource(code);
        void NotebookActions.run(notebook, panel.sessionContext);
      }
    };

    // Status bar item showing the current active DataFrame
    if (statusBar) {
      const statusItem = new Widget();
      statusItem.addClass('pv-statusbar-item');
      statusItem.node.textContent = '';
      explorer.setCurrentTableChangedCallback(name => {
        statusItem.node.textContent = name ? `df: ${name}` : '';
      });
      statusBar.registerStatusItem('@pivotal/jupyterlab:current-table', {
        item: statusItem,
        align: 'left',
        rank: 100,
      });
    }

    // Clicking an explorer item focuses the viewer on that item
    explorer.setItemClickCallback(name => {
      viewer.focusItem(name);
      activateViewer();
    });

    // Deleting from explorer also removes from viewer and Python namespace
    explorer.setDeleteCallback(name => viewer.requestDelete(name));

    // Show explorer alongside the viewer whenever the viewer is activated.
    // Left and right sidebars are independent — activating the explorer (left)
    // does not deactivate the viewer (right), so no RAF re-activation needed.
    viewer.setActivateCallback(() => {
      app.shell.activateById(explorer.id);
    });

    // Register commands
    app.commands.addCommand('pivotal:show-viewer', {
      label: 'Show Pivotal Viewer',
      execute: () => {
        activateViewer();
        setTimeout(() => viewer.node.focus(), 100);
      },
    });

    app.commands.addCommand('pivotal:show-explorer', {
      label: 'Show Pivotal Explorer',
      execute: () => {
        app.shell.activateById(explorer.id);
        setTimeout(() => explorer.node.focus(), 100);
      },
    });

    app.commands.addCommand('pivotal:viewer-back', {
      label: 'Pivotal Viewer: Back',
      execute: () => viewer.back(),
    });

    app.commands.addCommand('pivotal:viewer-forward', {
      label: 'Pivotal Viewer: Forward',
      execute: () => viewer.forward(),
    });

    // New Pivotal cell command
    const insertNewPivotalCell = (panel: any) => {
      const notebook = panel.content;
      NotebookActions.insertBelow(notebook);
      const active = notebook.activeCell;
      if (active) {
        active.model.sharedModel.setSource('%%pivotal\n');
        notebook.mode = 'edit';
      }
    };

    app.commands.addCommand('pivotal:focus-notebook', {
      label: 'Focus Notebook',
      execute: () => {
        const panel = tracker.currentWidget;
        if (panel) {
          panel.content.node.focus();
          app.shell.activateById(panel.id);
        }
      },
    });

    app.commands.addCommand('pivotal:new-cell', {
      label: 'New Pivotal Cell',
      icon: pivotalGreyIcon,
      execute: () => {
        const panel = tracker.currentWidget;
        if (panel) insertNewPivotalCell(panel);
      },
    });

    // --- GUI cell commands ---
    app.commands.addCommand('pivotal:load-gui', {
      label: 'Load Data Source',
      execute: () => insertAndRun('import pivotal\npivotal.load_gui()'),
    });

    app.commands.addCommand('pivotal:pivot-gui', {
      label: 'Insert Pivot Table',
      execute: () => {
        const table = explorer.getCurrentTable();
        insertAndRun(table
          ? `import pivotal\npivotal.pivot_gui(${JSON.stringify(table)})`
          : `import pivotal\npivotal.pivot_gui()`);
      },
    });

    app.commands.addCommand('pivotal:plot-gui', {
      label: 'Insert Pivot Chart',
      execute: () => {
        const table = explorer.getCurrentTable();
        insertAndRun(table
          ? `import pivotal\npivotal.plot_gui(${JSON.stringify(table)})`
          : `import pivotal\npivotal.plot_gui()`);
      },
    });

    app.commands.addCommand('pivotal:save-gui', {
      label: 'Save Data Package',
      execute: () => {
        const table = explorer.getCurrentTable();
        insertAndRun(table
          ? `import pivotal\npivotal.save_gui(${JSON.stringify(table)})`
          : `import pivotal\npivotal.save_gui()`);
      },
    });

    app.commands.addCommand('pivotal:settings-gui', {
      label: 'Settings',
      execute: () => insertAndRun('import pivotal\npivotal.settings_gui()'),
    });

    app.commands.addCommand('pivotal:export-py', {
      label: 'Generate Python Code',
      execute: async () => {
        const panel = tracker.currentWidget;
        if (!panel) return;
        const relPath = panel.context.path;
        const kernel = panel.context.sessionContext.session?.kernel;
        if (!kernel) { Notification.emit('No active kernel — run a cell first.', 'warning'); return; }

        // Construct absolute path using the JupyterLab server root
        const serverRoot = PageConfig.getOption('serverRoot') || PageConfig.getOption('rootUri') || '';
        const absPath = serverRoot ? `${serverRoot}/${relPath}` : relPath;

        const code = [
          'from pivotal.__main__ import notebook_to_python as _ntp',
          `_ntp(${JSON.stringify(absPath)})`,
        ].join('\n');

        let errorText = '';
        const future = kernel.requestExecute({ code });
        future.onIOPub = (msg: any) => {
          if (msg.msg_type === 'stream') {
            console.log('[pivotal export]', msg.content?.text?.trim());
          } else if (msg.msg_type === 'error') {
            errorText = msg.content?.ename + ': ' + msg.content?.evalue;
            console.error('[pivotal export error]', errorText);
          }
        };
        await future.done;
        if (errorText) {
          Notification.emit(`Export failed: ${errorText}`, 'error', { autoClose: false });
        } else {
          Notification.emit(`Exported: ${relPath.replace(/\.ipynb$/, '.py')}`, 'success', { autoClose: 5000 });
        }
      },
    });

    // Keyboard shortcuts — Alt+key works in all modes; chords only in command mode
    const CMD = '.jp-Notebook.jp-mod-commandMode';

    app.commands.addKeyBinding({ command: 'pivotal:new-cell',      keys: ['Alt P'],    selector: 'body' });
    app.commands.addKeyBinding({ command: 'pivotal:new-cell',      keys: ['P', 'P'],   selector: CMD    });
    app.commands.addKeyBinding({ command: 'pivotal:show-viewer',   keys: ['Alt V'],    selector: 'body' });
    app.commands.addKeyBinding({ command: 'pivotal:show-viewer',   keys: ['V', 'V'],   selector: CMD    });
    app.commands.addKeyBinding({ command: 'pivotal:show-explorer', keys: ['Alt E'],    selector: 'body' });
    app.commands.addKeyBinding({ command: 'pivotal:show-explorer', keys: ['E', 'E'],   selector: CMD    });
    app.commands.addKeyBinding({ command: 'pivotal:focus-notebook', keys: ['Alt N'],    selector: 'body' });
    app.commands.addKeyBinding({ command: 'pivotal:focus-notebook', keys: ['N', 'N'],   selector: CMD    });
    app.commands.addKeyBinding({ command: 'pivotal:viewer-back',   keys: ['Alt ['],    selector: 'body' });
    app.commands.addKeyBinding({ command: 'pivotal:viewer-forward',keys: ['Alt ]'],    selector: 'body' });

    // Add expanded Pivotal dropdown menu to every notebook toolbar
    tracker.widgetAdded.connect((_, panel) => {
      const menu = new Menu({ commands: app.commands });
      menu.addItem({ command: 'pivotal:new-cell' });
      menu.addItem({ type: 'separator' });
      menu.addItem({ command: 'pivotal:load-gui' });
      menu.addItem({ command: 'pivotal:pivot-gui' });
      menu.addItem({ command: 'pivotal:plot-gui' });
      menu.addItem({ command: 'pivotal:save-gui' });
      menu.addItem({ type: 'separator' });
      menu.addItem({ command: 'pivotal:settings-gui' });
      menu.addItem({ type: 'separator' });
      menu.addItem({ command: 'pivotal:export-py' });

      const btn = new ToolbarButton({
        icon: pivotalGreyIcon,
        tooltip: 'Pivotal',
        onClick: () => {
          const rect = btn.node.getBoundingClientRect();
          menu.open(rect.left, rect.bottom);
        },
      });
      panel.toolbar.addItem('pivotal-menu', btn);
    });

    // Registry mapping gui_id → cell model for upsert behaviour
    const _cellRegistry = new Map<string, any>();

    // Wire comm target whenever a kernel is connected
    function registerCommOnKernel(kernel: { registerCommTarget?: Function } | null | undefined) {
      if (!kernel || typeof kernel.registerCommTarget !== 'function') return;
      kernel.registerCommTarget('pivotal_viewer', (comm: any) => {
        viewer.setComm(comm);
        comm.onMsg = (msg: any) => {
          const data = msg?.content?.data as (ViewerMessage & { name?: string }) | undefined;
          if (data?.type === 'dataframe' || data?.type === 'chart' || data?.type === 'gt_table') {
            viewer.push(data as ViewerMessage);
            // Show panel when new data arrives
            activateViewer();
          } else if ((data as any)?.type === 'current_table') {
            explorer.setCurrentTable((data as any).name ?? null);
          } else if ((data as any)?.type === 'delete') {
            // Python-initiated delete: remove from viewer only (Python already removed from namespace)
            viewer.deleteItem((data as any).name ?? '');
          } else if ((data as any)?.type === 'insert_cell') {
            // Load GUI: always insert a fresh cell
            const panel = tracker.currentWidget;
            if (panel) {
              const notebook = panel.content;
              const code = (data as any).code as string ?? '';
              NotebookActions.insertBelow(notebook);
              const active = notebook.activeCell;
              if (active) {
                active.model.sharedModel.setSource(code);
                void NotebookActions.run(notebook, panel.sessionContext);
              }
            }
          } else if ((data as any)?.type === 'upsert_cell') {
            // Plot/pivot GUI: update the previously inserted cell, or insert if first time
            const panel = tracker.currentWidget;
            if (panel) {
              const notebook = panel.content;
              const code   = (data as any).code   as string ?? '';
              const guiId  = (data as any).gui_id as string ?? '';
              const cells  = notebook.model?.cells;
              let cellIdx  = -1;
              if (guiId && cells) {
                const existing = _cellRegistry.get(guiId);
                if (existing) {
                  for (let i = 0; i < cells.length; i++) {
                    if (cells.get(i) === existing) { cellIdx = i; break; }
                  }
                }
              }
              if (cellIdx >= 0) {
                // Update existing cell and re-run it
                const existing = _cellRegistry.get(guiId);
                existing.sharedModel.setSource(code);
                notebook.activeCellIndex = cellIdx;
                void NotebookActions.run(notebook, panel.sessionContext);
              } else {
                // First time: insert a new cell below and register it
                NotebookActions.insertBelow(notebook);
                const active = notebook.activeCell;
                if (active) {
                  active.model.sharedModel.setSource(code);
                  if (guiId) _cellRegistry.set(guiId, active.model);
                  void NotebookActions.run(notebook, panel.sessionContext);
                }
              }
            }
          }
        };
      });
    }

    // Register on existing notebook
    const attachToPanel = (panel: any) => {
      if (!panel) return;
      const ctx = panel.sessionContext;
      if (!ctx) return;
      registerCommOnKernel(ctx.session?.kernel);
      ctx.kernelChanged.connect((_: any, args: any) => {
        // Kernel replaced (restart or new kernel) — clear stale viewer content
        viewer.clear();
        if (args.newValue) registerCommOnKernel(args.newValue);
      });
      ctx.statusChanged.connect((_: any, status: string) => {
        if (status === 'restarting' || status === 'autorestarting' ||
            status === 'terminating' || status === 'dead') {
          viewer.clear();
        }
      });
      // Clear when the notebook is closed
      panel.disposed.connect(() => viewer.clear());
    };

    tracker.currentChanged.connect((_, panel) => attachToPanel(panel));
    tracker.forEach(panel => attachToPanel(panel));
  },
};

export default [plugin, viewerPlugin];
