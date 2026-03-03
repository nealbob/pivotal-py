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
  autocompletion,
  CompletionContext,
  CompletionResult,
  Completion,
} from '@codemirror/autocomplete';

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
                  ? Prec.highest(new LanguageSupport(pivotalLanguage, completionExt))
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

export default plugin;
