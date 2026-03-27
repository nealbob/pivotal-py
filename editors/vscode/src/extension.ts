import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';

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

// ---------------------------------------------------------------------------
// Autocomplete file loading — mtime-cached so each keystroke is free
// ---------------------------------------------------------------------------

let _acCache: { filePath: string; mtime: number; data: AutocompleteData } | null = null;

function findAutocompleteFilePath(documentUri: vscode.Uri): string | null {
  const candidates: string[] = [
    path.join(path.dirname(documentUri.fsPath), 'pivotal_autocomplete.json'),
  ];
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    const p = path.join(folder.uri.fsPath, 'pivotal_autocomplete.json');
    if (!candidates.includes(p)) candidates.push(p);
  }
  for (const p of candidates) {
    try { fs.statSync(p); return p; } catch { /* not found */ }
  }
  return null;
}

function loadAutocompleteFile(documentUri: vscode.Uri): AutocompleteData | null {
  const filePath = findAutocompleteFilePath(documentUri);
  if (!filePath) return null;
  try {
    const mtime = fs.statSync(filePath).mtimeMs;
    if (_acCache && _acCache.filePath === filePath && _acCache.mtime === mtime) {
      return _acCache.data;
    }
    const data: AutocompleteData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    _acCache = { filePath, mtime, data };
    return data;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Context detection
// ---------------------------------------------------------------------------

const COMMAND_KEYWORDS = [
  'df', 'load', 'filter', 'select', 'sort', 'assign', 'group by',
  'merge', 'left merge', 'right merge', 'inner merge', 'outer merge',
  'concat', 'intersect', 'exclude', 'pivot', 'plot', 'drop', 'rename', 'fillna', 'dropna',
  'distinct', 'python', 'save', 'apply', 'summarise',
];

const AGG_KEYWORDS = ['mean', 'sum', 'count', 'min', 'max', 'median', 'std', 'avg'];
const CHART_TYPES = ['line', 'bar', 'scatter', 'hist', 'box', 'area'];

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

  // Empty root-level line → command keywords
  if (trimmed === '' && indent === 0) return { type: 'command' };

  // `df <word>` or `df <word> from <word>` → table names
  if (/^df\s+\w*$/.test(trimmed)) return { type: 'table' };
  if (/^df\s+\w+\s+from\s+\w*$/.test(trimmed)) return { type: 'table' };

  // After merge / concat → table names
  if (/^(left\s+|right\s+|inner\s+|outer\s+)?(merge|concat|intersect|exclude)\s+\w*$/.test(trimmed)) {
    return { type: 'table' };
  }

  // After `plot` (first arg is chart kind or name) → chart types
  if (/^plot\s+\w*$/.test(trimmed)) return { type: 'charttype' };

  // After `agg` → aggregation functions; after `agg <func>` → columns
  if (/^agg\s+\w*$/.test(trimmed)) return { type: 'agg' };
  if (/^agg\s+\w+\s+\w*$/.test(trimmed)) {
    const table = findActiveTable(lines, cursorLine, ac);
    return table ? { type: 'column', table } : { type: 'none' };
  }

  // Column contexts — need an active table
  const table = findActiveTable(lines, cursorLine, ac);
  if (table) {
    // filter / select / drop / sort / distinct / rename (first arg onwards)
    if (/^(filter|select|drop|distinct|sort|rename)\b/.test(trimmed)) {
      return { type: 'column', table };
    }
    // assign — after `=`
    if (/^assign\s+\w+\s*=/.test(trimmed)) {
      return { type: 'column', table };
    }
    // group by / by (inside pivot)
    if (/^(group\s+by|by)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
    // plot sub-params: x, y, by, c (column axis selectors)
    if (/^(x|y|by|c)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
  }

  // Partial keyword at root level → command keywords
  return { type: 'command' };
}

// ---------------------------------------------------------------------------
// Build VS Code completion items
// ---------------------------------------------------------------------------

function buildItems(ctx: CompletionCtx, ac: AutocompleteData | null): vscode.CompletionItem[] {
  switch (ctx.type) {
    case 'command':
      return COMMAND_KEYWORDS.map(kw =>
        new vscode.CompletionItem(kw, vscode.CompletionItemKind.Keyword)
      );

    case 'table':
      if (!ac) return [];
      return Object.keys(ac.tables).map(t =>
        new vscode.CompletionItem(t, vscode.CompletionItemKind.Variable)
      );

    case 'column': {
      if (!ac) return [];
      const info = ac.tables[ctx.table];
      if (!info) return [];
      return info.columns.map(col => {
        const label = Array.isArray(col) ? col.join('.') : String(col);
        const item = new vscode.CompletionItem(label, vscode.CompletionItemKind.Field);
        const dtype = info.dtypes?.[label];
        if (dtype) item.detail = dtype;
        return item;
      });
    }

    case 'agg':
      return AGG_KEYWORDS.map(kw =>
        new vscode.CompletionItem(kw, vscode.CompletionItemKind.Function)
      );

    case 'charttype':
      return CHART_TYPES.map(ct =>
        new vscode.CompletionItem(ct, vscode.CompletionItemKind.EnumMember)
      );

    case 'none':
      return [];
  }
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {

  // --- Command: Execute File via CLI ---
  const executeFile = vscode.commands.registerCommand('pivotal.executeFile', () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { vscode.window.showErrorMessage('Pivotal: No active editor.'); return; }
    const filePath = editor.document.uri.fsPath;
    if (!filePath.endsWith('.pivotal')) {
      vscode.window.showErrorMessage('Pivotal: Active file is not a .pivotal file.');
      return;
    }
    editor.document.save().then(() => {
      const terminal = vscode.window.createTerminal('Pivotal');
      terminal.show(true);
      terminal.sendText(`python -m pivotal "${filePath}"`);
    });
  });

  // --- Command: Execute File in Interactive Notebook ---
  const executeInNotebook = vscode.commands.registerCommand(
    'pivotal.executeInNotebook',
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) { vscode.window.showErrorMessage('Pivotal: No active editor.'); return; }
      const filePath = editor.document.uri.fsPath;
      if (!filePath.endsWith('.pivotal')) {
        vscode.window.showErrorMessage('Pivotal: Active file is not a .pivotal file.');
        return;
      }
      await editor.document.save();
      const fileContents = fs.readFileSync(filePath, 'utf8');
      const sections = fileContents.split(/^#%%[^\n]*$/m).map(s => s.trim()).filter(Boolean);
      for (const section of sections) {
        const escapedContents = JSON.stringify(section);
        const cellText =
          `import pivotal; get_ipython().run_cell_magic('pivotal', '', ${escapedContents})`;
        try {
          await vscode.commands.executeCommand('jupyter.execSelectionInteractive', cellText);
        } catch {
          vscode.window.showErrorMessage(
            'Pivotal: Failed to send code to Interactive Window. ' +
            'Please ensure a Python kernel is running.'
          );
          return;
        }
        if (sections.length > 1) {
          await new Promise(resolve => setTimeout(resolve, 300));
        }
      }
    }
  );

  // --- Command: Execute Selection in Interactive Notebook ---
  const executeSelectionInNotebook = vscode.commands.registerCommand(
    'pivotal.executeSelectionInNotebook',
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) { vscode.window.showErrorMessage('Pivotal: No active editor.'); return; }
      const selection = editor.selection;
      const selectedText = editor.document.getText(selection);
      if (!selectedText.trim()) {
        vscode.window.showInformationMessage('Pivotal: No text selected.');
        return;
      }
      const escapedContents = JSON.stringify(selectedText);
      const cellText =
        `import pivotal; get_ipython().run_cell_magic('pivotal', '', ${escapedContents})`;
      try {
        await vscode.commands.executeCommand('jupyter.execSelectionInteractive', cellText);
      } catch {
        vscode.window.showErrorMessage(
          'Pivotal: Failed to send selection to Interactive Window. ' +
          'Please ensure a Python kernel is running.'
        );
      }
    }
  );

  // --- Command: Compile .pivotal file to Python ---
  const compileToFile = vscode.commands.registerCommand('pivotal.compileToFile', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { vscode.window.showErrorMessage('Pivotal: No active editor.'); return; }
    const filePath = editor.document.uri.fsPath;
    if (!filePath.endsWith('.pivotal')) {
      vscode.window.showErrorMessage('Pivotal: Active file is not a .pivotal file.');
      return;
    }
    await editor.document.save();
    const pyPath = filePath.replace(/\.pivotal$/, '.py');
    return new Promise<void>(resolve => {
      exec(`python -m pivotal --compile "${filePath}"`, (error, _stdout, stderr) => {
        if (error) {
          vscode.window.showErrorMessage(`Pivotal compile error: ${stderr || error.message}`);
        } else {
          vscode.window.showInformationMessage(`Pivotal: Compiled to ${pyPath}`, 'Open File')
            .then(action => {
              if (action === 'Open File') {
                vscode.workspace.openTextDocument(pyPath).then(doc =>
                  vscode.window.showTextDocument(doc)
                );
              }
            });
        }
        resolve();
      });
    });
  });

  // --- Completion provider for .pivotal files ---
  const completionProvider = vscode.languages.registerCompletionItemProvider(
    { language: 'pivotal' },
    {
      provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
      ): vscode.CompletionItem[] {
        const ac = loadAutocompleteFile(document.uri);
        const lines = document.getText().split('\n');
        const ctx = detectContext(lines, position.line, position.character, ac);
        return buildItems(ctx, ac);
      },
    },
    ' ', '\t',
  );

  context.subscriptions.push(
    executeFile,
    executeInNotebook,
    executeSelectionInNotebook,
    compileToFile,
    completionProvider,
  );
}

export function deactivate(): void { /* nothing to clean up */ }
