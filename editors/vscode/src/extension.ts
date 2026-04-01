import * as vscode from 'vscode';
import * as fs from 'fs';
import * as net from 'net';
import * as os from 'os';
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

interface CommandCompletion {
  label: string;
  snippet?: string; // VS Code snippet string (${1:placeholder} syntax)
  detail?: string;
}

const COMMAND_COMPLETIONS: CommandCompletion[] = [
  // Table declarations
  { label: 'df',     detail: 'df <table> [from <source>]' },
  { label: 'load',   snippet: 'load ${1:tbl} ${2:path}',   detail: 'load <table> <path>' },

  // Row operations
  { label: 'filter',   detail: 'filter <condition>' },
  { label: 'select',   detail: 'select <col>, ...' },
  { label: 'drop',     detail: 'drop <col>, ...' },
  { label: 'distinct', detail: 'distinct [<col>, ...]' },
  { label: 'sort',     detail: 'sort <col> [asc|desc]' },

  // Merging / set ops
  { label: 'merge',       snippet: 'merge ${1:tbl} on ${2:key}',       detail: 'merge <table> on <key>' },
  { label: 'left merge',  snippet: 'left merge ${1:tbl} on ${2:key}',  detail: 'left merge <table> on <key>' },
  { label: 'right merge', snippet: 'right merge ${1:tbl} on ${2:key}', detail: 'right merge <table> on <key>' },
  { label: 'inner merge', snippet: 'inner merge ${1:tbl} on ${2:key}', detail: 'inner merge <table> on <key>' },
  { label: 'outer merge', snippet: 'outer merge ${1:tbl} on ${2:key}', detail: 'outer merge <table> on <key>' },
  { label: 'concat',    detail: 'concat <table>, ...' },
  { label: 'intersect', detail: 'intersect <table>' },
  { label: 'exclude',   detail: 'exclude <table>' },

  // Grouping / aggregation
  { label: 'group by', snippet: 'group by ${1:grp_col}\n    agg ${2:func} ${3:val_col} as ${4:name}', detail: 'group by <col>\n    agg <func> <col> as <name>' },
  { label: 'agg',      snippet: 'agg ${1:func} ${2:agg_col} as ${3:name}', detail: 'agg <func> <col> as <name>' },

  // Window functions
  { label: 'rolling', snippet: 'rolling ${1:func} ${2:val_col} ${3:window} as ${4:name}', detail: 'rolling <func> <col> <window> as <name>' },
  { label: 'rank',    snippet: 'rank ${1:rank_col} as ${2:name}',               detail: 'rank <col> as <name>' },
  { label: 'lag',     snippet: 'lag ${1:val_col} ${2:n} as ${3:name}',          detail: 'lag <col> <n> as <name>' },
  { label: 'lead',    snippet: 'lead ${1:val_col} ${2:n} as ${3:name}',         detail: 'lead <col> <n> as <name>' },
  { label: 'cumsum',  snippet: 'cumsum ${1:val_col} as ${2:name}',              detail: 'cumsum <col> as <name>' },
  { label: 'cummean', snippet: 'cummean ${1:val_col} as ${2:name}',             detail: 'cummean <col> as <name>' },
  { label: 'cummin',  snippet: 'cummin ${1:val_col} as ${2:name}',              detail: 'cummin <col> as <name>' },
  { label: 'cummax',  snippet: 'cummax ${1:val_col} as ${2:name}',              detail: 'cummax <col> as <name>' },

  // Reshaping
  { label: 'pivot',   snippet: 'pivot\n    rows ${1:row_col}\n    cols ${2:hdr_col}\n    agg ${3:func} ${4:val_col} as ${5:name}', detail: 'pivot\n    rows <col>  cols <col>  agg <func> <col> as <name>' },
  { label: 'unpivot', snippet: 'unpivot\n    cols ${1:col1}, ${2:col2}\n    id ${3:id_col}', detail: 'unpivot\n    cols <col>, ...  id <id_col>' },

  // Type casting
  { label: 'cast', snippet: 'cast ${1:cast_col} as ${2:type}', detail: 'cast <col> as <type> [strict]' },

  // Filling / cleaning
  { label: 'fillna',  detail: 'fillna <value>' },
  { label: 'dropna',  detail: 'dropna [<col>, ...]' },
  { label: 'rename',  snippet: 'rename ${1:old_col} as ${2:new_col}', detail: 'rename <col> as <new_name>' },

  // Plotting
  { label: 'plot',     snippet: 'plot ${1:line}\n    x ${2:x_col}\n    y ${3:y_col}',         detail: 'plot <type>  x <col>  y <col>' },
  { label: 'agg plot', snippet: 'agg plot ${1:bar}\n    x ${2:x_col}\n    y ${3:y_col}',      detail: 'agg plot <type>  x <col>  y <col>' },

  // Output / misc
  { label: 'show' },
  { label: 'show head',    detail: 'show head [<n>]' },
  { label: 'show summary', detail: 'show summary' },
  { label: 'save',   snippet: 'save "${1:path}"', detail: 'save "<path>"' },
  { label: 'apply',  detail: 'apply <func>' },
  { label: 'table',  detail: 'table' },
  { label: 'delete', detail: 'delete <table>' },
  { label: 'python', detail: 'python' },
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

  if (trimmed === '' && indent === 0) return { type: 'command' };

  if (/^df\s+\w*$/.test(trimmed)) return { type: 'table' };
  if (/^df\s+\w+\s+from\s+\w*$/.test(trimmed)) return { type: 'table' };
  if (/^(left\s+|right\s+|inner\s+|outer\s+)?(merge|concat|intersect|exclude)\s+\w*$/.test(trimmed)) {
    return { type: 'table' };
  }
  if (/^(left\s+|right\s+|inner\s+|outer\s+)?merge\s+\w+\s+on\s+\w*$/.test(trimmed)) {
    const table = findActiveTable(lines, cursorLine, ac);
    return table ? { type: 'column', table } : { type: 'none' };
  }

  if (/^(agg\s+)?plot\s+\w*$/.test(trimmed)) return { type: 'charttype' };

  if (/^agg\s+\w*$/.test(trimmed)) return { type: 'agg' };
  if (/^agg\s+\w+\s+\w*$/.test(trimmed)) {
    const table = findActiveTable(lines, cursorLine, ac);
    return table ? { type: 'column', table } : { type: 'none' };
  }

  if (/^rolling\s+\w*$/.test(trimmed)) return { type: 'agg' };
  if (/^rolling\s+\w+\s+\w*$/.test(trimmed)) {
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
    if (/^(rows|cols|id|order|stub|left_on|right_on)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^agg\s+(mean|sum|count|min|max|avg|median|std|nunique|wavg)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^(mean|sum|count|min|max|avg|median|std|nunique|wavg)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
  }

  if (indent > 0 && /\s/.test(trimmed)) {
    return { type: 'none' };
  }

  return { type: 'command' };
}

// ---------------------------------------------------------------------------
// Build VS Code completion items
// ---------------------------------------------------------------------------

function buildItems(ctx: CompletionCtx, ac: AutocompleteData | null): vscode.CompletionItem[] {
  switch (ctx.type) {
    case 'command':
      return COMMAND_COMPLETIONS.map(({ label, snippet, detail }) => {
        const item = new vscode.CompletionItem(label, vscode.CompletionItemKind.Keyword);
        if (detail) item.detail = detail;
        if (snippet) item.insertText = new vscode.SnippetString(snippet);
        return item;
      });

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
// Hover — show syntax signature for the command on the current line
// ---------------------------------------------------------------------------

function hoverForLine(line: string): vscode.Hover | undefined {
  const trimmed = line.trimStart();

  // Find the longest COMMAND_COMPLETIONS label that is a prefix of the trimmed
  // line and is followed by whitespace or end-of-line (full-word boundary).
  let best: CommandCompletion | undefined;
  for (const cmd of COMMAND_COMPLETIONS) {
    if (!cmd.detail) continue;
    if (!trimmed.startsWith(cmd.label)) continue;
    const after = trimmed[cmd.label.length];
    if (after !== undefined && after !== ' ' && after !== '\t') continue;
    if (!best || cmd.label.length > best.label.length) best = cmd;
  }

  if (!best?.detail) return undefined;

  const md = new vscode.MarkdownString();
  md.appendCodeblock(best.detail, 'pivotal');
  return new vscode.Hover(md);
}

// ---------------------------------------------------------------------------
// VS Code Bridge — TCP connection to the Python magic layer
// ---------------------------------------------------------------------------

const BRIDGE_FILE = path.join(os.tmpdir(), 'pivotal_bridge.json');

// Registered handlers receive every inbound message from Python.
// Phase 3/4 (viewer, explorer) will push handlers here.
type BridgeMessageHandler = (msg: Record<string, unknown>) => void;
const _bridgeHandlers: BridgeMessageHandler[] = [];

let _bridgeSocket: net.Socket | null = null;
let _bridgeStatusBar: vscode.StatusBarItem | null = null;

/** Register a handler for messages arriving from the Python bridge. */
export function onBridgeMessage(handler: BridgeMessageHandler): void {
  _bridgeHandlers.push(handler);
}

/** Send a message to the Python bridge (fire-and-forget). */
export function sendToBridge(data: Record<string, unknown>): void {
  if (!_bridgeSocket) { return; }
  try {
    _bridgeSocket.write(JSON.stringify(data) + '\n');
  } catch {
    _bridgeSocket = null;
  }
}

function _setBridgeStatus(state: 'connected' | 'waiting' | 'disconnected'): void {
  if (!_bridgeStatusBar) { return; }
  switch (state) {
    case 'connected':
      _bridgeStatusBar.text = '$(circle-filled) Pivotal';
      _bridgeStatusBar.tooltip = 'Pivotal bridge connected';
      _bridgeStatusBar.backgroundColor = undefined;
      break;
    case 'waiting':
      _bridgeStatusBar.text = '$(circle-outline) Pivotal';
      _bridgeStatusBar.tooltip = 'Pivotal: waiting for Python kernel';
      _bridgeStatusBar.backgroundColor = undefined;
      break;
    case 'disconnected':
      _bridgeStatusBar.text = '$(warning) Pivotal';
      _bridgeStatusBar.tooltip = 'Pivotal bridge disconnected — restart kernel to reconnect';
      _bridgeStatusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
      break;
  }
  _bridgeStatusBar.show();
}

function _connectBridge(port: number): void {
  // Tear down any existing socket before reconnecting.
  if (_bridgeSocket) {
    _bridgeSocket.destroy();
    _bridgeSocket = null;
  }

  const socket = net.createConnection(port, '127.0.0.1');
  let recvBuf = '';

  socket.on('connect', () => {
    _bridgeSocket = socket;
    _setBridgeStatus('connected');
  });

  socket.on('data', (chunk: Buffer) => {
    recvBuf += chunk.toString();
    const lines = recvBuf.split('\n');
    recvBuf = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) { continue; }
      try {
        const msg = JSON.parse(trimmed) as Record<string, unknown>;
        for (const handler of _bridgeHandlers) {
          try { handler(msg); } catch { /* isolate handler errors */ }
        }
      } catch { /* malformed JSON — ignore */ }
    }
  });

  socket.on('close', () => {
    if (_bridgeSocket === socket) { _bridgeSocket = null; }
    _setBridgeStatus('waiting');
  });

  socket.on('error', () => {
    socket.destroy();
    if (_bridgeSocket === socket) { _bridgeSocket = null; }
    _setBridgeStatus('disconnected');
  });
}

function _tryReadBridgeFile(): void {
  try {
    const raw = fs.readFileSync(BRIDGE_FILE, 'utf-8');
    const info = JSON.parse(raw) as { port?: number; pid?: number };
    if (typeof info.port === 'number') {
      _connectBridge(info.port);
    }
  } catch { /* file not present yet */ }
}

function _startBridgeWatcher(context: vscode.ExtensionContext): void {
  _bridgeStatusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 10);
  _setBridgeStatus('waiting');
  context.subscriptions.push(_bridgeStatusBar);

  // Attempt immediate connection if the bridge file already exists
  // (kernel was running before VS Code opened this workspace).
  _tryReadBridgeFile();

  // Watch the temp directory for bridge file creation / updates.
  // fs.watch fires on both create and change events.
  try {
    const watcher = fs.watch(os.tmpdir(), (_event, filename) => {
      if (filename === 'pivotal_bridge.json') {
        // Small delay to let Python finish the atomic rename.
        setTimeout(_tryReadBridgeFile, 100);
      }
    });
    context.subscriptions.push({ dispose: () => watcher.close() });
  } catch { /* fs.watch not available on this platform */ }
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

  // --- Hover provider for .pivotal files ---
  const hoverProvider = vscode.languages.registerHoverProvider(
    { language: 'pivotal' },
    {
      provideHover(document: vscode.TextDocument, position: vscode.Position) {
        const line = document.lineAt(position.line).text;
        return hoverForLine(line);
      },
    },
  );

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

  _startBridgeWatcher(context);

  context.subscriptions.push(
    executeFile,
    executeInNotebook,
    executeSelectionInNotebook,
    compileToFile,
    hoverProvider,
    completionProvider,
  );
}

export function deactivate(): void { /* nothing to clean up */ }
