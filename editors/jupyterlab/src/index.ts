import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
} from '@jupyterlab/application';

import { LabIcon } from '@jupyterlab/ui-components';

import {
  IEditorLanguageRegistry,
  IEditorExtensionRegistry,
} from '@jupyterlab/codemirror';
import { LanguageSupport } from '@codemirror/language';
import { Compartment, Prec } from '@codemirror/state';
import { EditorView } from '@codemirror/view';

import {
  ICompletionProviderManager,
  ICompletionProvider,
  ICompletionContext,
  CompletionHandler,
} from '@jupyterlab/completer';

import { pivotalLanguage } from './language';

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
  'df', 'load', 'filter', 'select', 'sort', 'assign', 'group by',
  'merge', 'left merge', 'right merge', 'inner merge', 'outer merge',
  'concat', 'pivot', 'plot', 'drop', 'rename', 'fillna', 'dropna',
  'distinct', 'python', 'save', 'apply',
];

const AGG_KEYWORDS = ['mean', 'sum', 'count', 'min', 'max', 'median', 'std', 'avg'];
const CHART_TYPES = ['line', 'bar', 'scatter', 'hist', 'box', 'area'];

// ---------------------------------------------------------------------------
// Autocomplete file fetching — last_modified-cached
// ---------------------------------------------------------------------------

let _acCache: { path: string; lastModified: string; data: AutocompleteData } | null = null;

async function fetchAutocompleteData(dir: string): Promise<AutocompleteData | null> {
  const acPath = dir ? `${dir}/.pivotal_autocomplete.json` : '.pivotal_autocomplete.json';
  try {
    const resp = await fetch(`/api/contents/${acPath}`);
    if (!resp.ok) return null;
    const json = await resp.json() as { content: string; last_modified: string };
    const lastModified = json.last_modified ?? '';
    if (_acCache && _acCache.path === acPath && _acCache.lastModified === lastModified) {
      return _acCache.data;
    }
    const data = JSON.parse(json.content) as AutocompleteData;
    _acCache = { path: acPath, lastModified, data };
    return data;
  } catch {
    return null;
  }
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
    if (/^assign\s+\w+\s*=/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^(group\s+by|by)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^(x|y|by|c)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
  }

  if (indent === 0) return { type: 'command' };
  return { type: 'none' };
}

// ---------------------------------------------------------------------------
// Build completion items (JupyterLab format)
// ---------------------------------------------------------------------------

function buildItems(
  ctx: CompletionCtx,
  ac: AutocompleteData | null,
): CompletionHandler.ICompletionItem[] {
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
        return { label, type: 'field', documentation: dtype };
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
// Helper: find word start/end around a character offset
// ---------------------------------------------------------------------------

function wordBounds(text: string, offset: number): { start: number; end: number } {
  let start = offset;
  while (start > 0 && /\w/.test(text[start - 1])) start--;
  let end = offset;
  while (end < text.length && /\w/.test(text[end])) end++;
  return { start, end };
}

// ---------------------------------------------------------------------------
// JupyterLab ICompletionProvider
// ---------------------------------------------------------------------------

class PivotalCompletionProvider implements ICompletionProvider {
  readonly identifier = 'pivotal-completer';

  async isApplicable(context: ICompletionContext): Promise<boolean> {
    const editor = context.editor;
    if (!editor) return false;
    const firstLine = editor.model.sharedModel.getSource().split('\n')[0] ?? '';
    return MAGIC_RE.test(firstLine);
  }

  async fetch(
    request: CompletionHandler.IRequest,
    context: ICompletionContext,
  ): Promise<CompletionHandler.ICompletionItemsReply> {
    const empty: CompletionHandler.ICompletionItemsReply = {
      start: request.offset, end: request.offset, items: [],
    };

    const { text, offset } = request;
    const allLines = text.split('\n');
    const lineOffset = MAGIC_RE.test(allLines[0] ?? '') ? 1 : 0;

    // Convert character offset to line/col
    let remaining = offset;
    let lineIdx = 0;
    for (let i = 0; i < allLines.length; i++) {
      if (remaining <= allLines[i].length) { lineIdx = i; break; }
      remaining -= allLines[i].length + 1; // +1 for \n
      lineIdx = i + 1;
    }
    const col = remaining;
    const cursorLine = lineIdx - lineOffset;

    if (cursorLine < 0) return empty;

    const effectiveLines = allLines.slice(lineOffset);
    const dir = this._notebookDir(context);
    const ac = await fetchAutocompleteData(dir);

    const ctx = detectContext(effectiveLines, cursorLine, col, ac);
    const items = buildItems(ctx, ac);

    if (!items.length) return empty;

    const { start, end } = wordBounds(text, offset);
    return { start, end, items };
  }

  shouldShowContinuousHint(): boolean {
    return true;
  }

  private _notebookDir(context: ICompletionContext): string {
    const widget = context.widget;
    if ('context' in widget) {
      const ctxPath = (widget as any).context?.path as string | undefined;
      if (ctxPath) {
        return ctxPath.includes('/')
          ? ctxPath.slice(0, ctxPath.lastIndexOf('/'))
          : '';
      }
    }
    return '';
  }
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@pivotal/jupyterlab:language',
  description: 'Syntax highlighting and autocomplete for the Pivotal data transformation DSL',
  autoStart: true,
  requires: [IEditorLanguageRegistry, IEditorExtensionRegistry, ICompletionProviderManager],
  activate: (
    app: JupyterFrontEnd,
    languages: IEditorLanguageRegistry,
    extensions: IEditorExtensionRegistry,
    completionManager: ICompletionProviderManager,
  ) => {
    // Register the file type
    app.docRegistry.addFileType({
      name: 'pivotal',
      displayName: 'Pivotal',
      extensions: ['.pivotal'],
      mimeTypes: ['text/x-pivotal'],
      icon: pivotalIcon,
      contentType: 'file',
      fileFormat: 'text',
    });

    // Register the language for standalone .pivotal files
    languages.addLanguage({
      name: 'pivotal',
      mime: 'text/x-pivotal',
      extensions: ['.pivotal'],
      load: async () => new LanguageSupport(pivotalLanguage),
    });

    // For notebook cells: switch to Pivotal highlighting when %%pivotal is detected
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
                isPivotal ? Prec.highest(new LanguageSupport(pivotalLanguage)) : []
              ),
            });
          }),
        ];

        return { instance: () => ext, reconfigure: () => null };
      },
    });

    // Register with JupyterLab's completion system (handles Tab / Ctrl+Space)
    completionManager.registerProvider(new PivotalCompletionProvider());
  },
};

export default plugin;
