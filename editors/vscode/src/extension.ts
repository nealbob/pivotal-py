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
  { label: 'with',   snippet: 'with ${1:source} as ${2:output}', detail: 'with <source> [as <output>]' },
  { label: 'load',   snippet: 'load ${1:path} as ${2:tbl}',   detail: 'load <path> as <table>' },
  { label: 'from',   snippet: 'from ${1:path}\n\tload ${2:table} as ${3:df}', detail: 'from <database>\n    load <table> as <df>\n    query "SELECT..." as <df>' },
  { label: 'list',   snippet: 'list ${1:name} = ${2:item1}, ${3:item2}', detail: 'list <name> = <item>, ...' },
  { label: 'function', snippet: 'function ${1:name}(${2:input}, ${3:output})\n    with ${2:input} as ${3:output}\n        ${4:statement}\n    return ${3:output}', detail: 'function <name>(...)' },
  { label: 'return', snippet: 'return ${1:output}', detail: 'return <table>[, ...]' },

  // Row operations
  { label: 'for',      snippet: 'for ${1:col} in ${2:col1}, ${3:col2}\n    ${1:col} = ${1:col} / ${4:denom}', detail: 'for <name> in <col>, ... or :cols' },
  { label: 'filter',   detail: 'filter <condition>' },
  { label: 'assert',   snippet: 'assert ${1:col} ${2|unique,not null,>= 0|}', detail: 'assert <condition> | assert <col> unique | assert <col> not null' },
  { label: 'check',    snippet: 'check ${1:col} ${2|unique,not null,>= 0|}',  detail: 'check <condition> | check <col> unique | check <col> not null' },
  { label: 'select',   detail: 'select <col>, ... | select matches "<regex>"' },
  { label: 'drop',     detail: 'drop <col>, ... | drop matches "<regex>"' },
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
  { label: 'round', snippet: 'round ${1:col} ${2:digits}', detail: 'round <col>, ... <digits> [as <name>]' },

  // Filling / cleaning
  { label: 'fillna',  detail: 'fillna <value>' },
  { label: 'dropna',  detail: 'dropna [<col>, ...]' },
  { label: 'rename',  snippet: 'rename ${1:old_col} as ${2:new_col}', detail: 'rename <col> as <new_name>' },

  // Plotting
  { label: 'plot',     snippet: 'plot ${1:line}\n    x ${2:x_col}\n    y ${3:y_col}',         detail: 'plot <type>  x <col>  y <col>' },
  { label: 'pivot plot', snippet: 'pivot plot ${1:bar}\n    x ${2:x_col}\n    y ${3:mean} ${4:y_col}',      detail: 'pivot plot <type>  x <col>  y <func> <col>, ...' },

  // Output / misc
  { label: 'show' },
  { label: 'show head',    detail: 'show head [<n>]' },
  { label: 'show summary', detail: 'show summary' },
  { label: 'show shape',   detail: 'show shape' },
  { label: 'show columns', detail: 'show columns' },
  { label: 'save',   snippet: 'save "${1:path}"', detail: 'save "<path>"' },
  { label: 'apply',  detail: 'apply <func>' },
  { label: 'table',  detail: 'table' },
  { label: 'delete', detail: 'delete <table>' },
  { label: 'python', detail: 'python' },
];

const AGG_KEYWORDS = ['mean', 'sum', 'count', 'min', 'max', 'median', 'std', 'avg', 'quantile', 'percentile'];
const CHART_TYPES = ['line', 'bar', 'scatter', 'hist', 'box', 'area'];

function findActiveTable(
  lines: string[],
  cursorLine: number,
  ac: AutocompleteData | null,
): string | null {
  for (let i = cursorLine; i >= 0; i--) {
    const t = lines[i].trimStart();
    const withM = t.match(/^with\s+(\w+)(?:\s+as\s+(\w+))?/);
    if (withM) return withM[2] ?? withM[1];
    const loadM = t.match(/^load\s+\S+\s+as\s+(\w+)/);
    if (loadM) return loadM[1];
    // from block: scan indented load/query lines for the last alias defined
    const fromM = t.match(/^from\s+/);
    if (fromM) {
      let lastAlias: string | null = null;
      for (let j = i + 1; j <= cursorLine; j++) {
        const inner = lines[j]?.trimStart() ?? '';
        const lm = inner.match(/^load\s+([\w,\s]+as\s+\w+)/);
        if (lm) {
          const aliases = [...lm[1].matchAll(/\w+\s+as\s+(\w+)/g)];
          if (aliases.length) lastAlias = aliases[aliases.length - 1][1];
        }
        const qm = inner.match(/^query\s+.*\s+as\s+(\w+)/);
        if (qm) lastAlias = qm[1];
      }
      if (lastAlias) return lastAlias;
    }
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

  if (/^with\s+\w*$/.test(trimmed)) return { type: 'table' };
  if (/^with\s+\w+\s+as\s+\w*$/.test(trimmed)) return { type: 'table' };

  const forHeaderM = trimmed.match(/^for\s+\w+\s+in\s*(.*)$/);
  if (forHeaderM && !forHeaderM[1].trimStart().startsWith(':')) {
    const table = findActiveTable(lines, cursorLine, ac);
    return table ? { type: 'column', table } : { type: 'none' };
  }

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
    if (/^(filter|assert|check|select|drop|distinct|sort|rename)\b/.test(trimmed)) {
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
    if (/^agg\s+(mean|sum|count|min|max|avg|median|std|nunique|quantile|percentile|wavg|wmean)\s+\w*$/.test(trimmed)) {
      return { type: 'column', table };
    }
    if (/^(mean|sum|count|min|max|avg|median|std|nunique|quantile|percentile|wavg|wmean)\s+\w*$/.test(trimmed)) {
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
let _bridgePort: number | null = null;   // port of the currently connected (or connecting) bridge
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

function _injectDefaultSettings(): void {
  const cfg = vscode.workspace.getConfiguration('pivotal');
  const viewer         = cfg.get<boolean>('viewer', true);
  const backend        = cfg.get<string>('backend', 'pandas');
  const viewerFont     = cfg.get<number>('viewerFont', 1.0);
  const viewerNumFmt   = cfg.get<number>('viewerNumFormat', 5);
  const line = `viewer=${viewer} backend=${backend} viewer_font=${viewerFont} viewer_num_format=${viewerNumFmt}`;
  const cell = `import pivotal; get_ipython().run_line_magic('pivotal_set', ${JSON.stringify(line)})`;
  vscode.commands.executeCommand('jupyter.execSelectionInteractive', cell).then(
    undefined, () => { /* interactive window not open yet — silently ignore */ }
  );
}

function _connectBridge(port: number): void {
  // Skip if already connecting or connected to this port.
  // _bridgePort is set immediately when we start connecting (before the socket
  // fires 'connect'), so checking port alone is sufficient to avoid racing
  // with an in-flight connection attempt.
  if (_bridgePort === port) { return; }

  // Tear down any existing socket before reconnecting.
  if (_bridgeSocket) {
    _bridgeSocket.destroy();
    _bridgeSocket = null;
  }
  _bridgePort = port;

  const socket = net.createConnection(port, '127.0.0.1');
  let recvBuf = '';

  socket.on('connect', () => {
    _bridgeSocket = socket;
    _setBridgeStatus('connected');
    // Delay settings injection slightly so pending bridge data (flushed
    // immediately on accept) is processed first — avoids the
    // jupyter.execSelectionInteractive call racing with viewer panel creation.
    setTimeout(_injectDefaultSettings, 200);
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
    if (_bridgePort === port) { _bridgePort = null; }
    _setBridgeStatus('waiting');
  });

  socket.on('error', () => {
    socket.destroy();
    if (_bridgeSocket === socket) { _bridgeSocket = null; }
    if (_bridgePort === port) { _bridgePort = null; }
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
  // On Linux/WSL2 the filename argument is not guaranteed to be non-null
  // (Node.js docs: "filename is not always guaranteed to be provided").
  // Guard against null and treat any change in the tmp dir as a potential
  // bridge file update.
  try {
    const watcher = fs.watch(os.tmpdir(), (_event, filename) => {
      if (!filename || filename === 'pivotal_bridge.json') {
        // Small delay to let Python finish the atomic rename.
        setTimeout(_tryReadBridgeFile, 100);
      }
    });
    context.subscriptions.push({ dispose: () => watcher.close() });
  } catch { /* fs.watch not available on this platform */ }

  // Polling fallback: check every 2 s in case fs.watch misses the event
  // (common on WSL2 and some Linux configurations).
  const poll = setInterval(_tryReadBridgeFile, 2000);
  context.subscriptions.push({ dispose: () => clearInterval(poll) });
}

/** Rapid bridge-file polling after cell execution.
 *  On the first execution the kernel starts, creates the bridge, and writes
 *  the bridge file.  On WSL2 `fs.watch` often misses this event, and the 2 s
 *  steady-state poll can feel sluggish.  This does a short burst of fast
 *  checks (every 300 ms for ~6 s) so the viewer connects promptly.  Stops
 *  early once the bridge is connected. */
let _rapidPollTimer: ReturnType<typeof setInterval> | null = null;
function _scheduleRapidBridgePoll(): void {
  if (_rapidPollTimer || _bridgeSocket) { return; }
  let remaining = 20;   // 20 × 300 ms = 6 s
  _rapidPollTimer = setInterval(() => {
    if (_bridgeSocket || --remaining <= 0) {
      clearInterval(_rapidPollTimer!);
      _rapidPollTimer = null;
      return;
    }
    _tryReadBridgeFile();
  }, 300);
}

// ---------------------------------------------------------------------------
// Phase 3 — Viewer WebviewPanel
// ---------------------------------------------------------------------------

let _viewerPanel: vscode.WebviewPanel | null = null;
let _viewerReady = false;   // true once the webview has confirmed 'ready'
let _viewerSplit = false;   // true once the viewer has been moved to a split

function _generateNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let nonce = '';
  for (let i = 0; i < 32; i++) { nonce += chars[Math.floor(Math.random() * chars.length)]; }
  return nonce;
}

function _getOrCreateViewerPanel(context: vscode.ExtensionContext, reveal = false): vscode.WebviewPanel {
  if (_viewerPanel) {
    // reveal=true: user explicitly opened the viewer — show it where it already is, no column move
    if (reveal) { _viewerPanel.reveal(undefined, false); }
    return _viewerPanel;
  }
  // Open in column 2 (where the notebook lives) WITHOUT stealing focus.
  // The split into a horizontal layout happens once the webview confirms
  // 'ready' — see the onDidReceiveMessage handler below.  Doing this at
  // creation time (with preserveFocus:false + moveEditorToAboveGroup)
  // caused races that dropped the first batch of bridge messages.
  const panel = vscode.window.createWebviewPanel(
    'pivotalViewer',
    'Pivotal Viewer',
    { viewColumn: vscode.ViewColumn.Two, preserveFocus: true },
    { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] },
  );
  panel.webview.html = _buildViewerHtml(panel.webview);
  panel.webview.onDidReceiveMessage((msg: Record<string, unknown>) => {
    if (msg.type === 'ready') {
      // Webview has finished loading — replay all items that may have been
      // sent (and lost) before the message listener was registered.
      _viewerReady = true;
      for (const [, payload] of _explorerItems) {
        panel.webview.postMessage(payload);
      }
      // Move the viewer into a horizontal split above the notebook — once only.
      if (!_viewerSplit) {
        _viewerSplit = true;
        panel.reveal(undefined, false);   // briefly focus the panel so the move targets it
        vscode.commands.executeCommand('workbench.action.moveEditorToAboveGroup').then(undefined, () => {});
      }
      return;
    }
    sendToBridge(msg);
    // Keep explorer in sync with deletions/clears initiated from the viewer
    if (msg.type === 'delete') {
      _explorerItems.delete(msg.name as string);
      _refreshExplorer();
    } else if (msg.type === 'clear') {
      _explorerItems.clear();
      _refreshExplorer();
    }
  }, undefined, context.subscriptions);
  panel.onDidDispose(() => { _viewerPanel = null; _viewerReady = false; _viewerSplit = false; }, undefined, context.subscriptions);
  _viewerPanel = panel;
  return panel;
}

function _buildViewerHtml(webview: vscode.Webview): string {
  const nonce = _generateNonce();
  // CSP: scripts only from nonce + CDN; frames allowed for GT table iframes (srcdoc)
  const csp = [
    `default-src 'none'`,
    `script-src 'nonce-${nonce}' https://cdn.jsdelivr.net`,
    `style-src 'unsafe-inline' https://cdn.jsdelivr.net`,
    `img-src data: blob:`,
    `frame-src *`,
  ].join('; ');

  const AG = 'https://cdn.jsdelivr.net/npm/ag-grid-community@33';

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<link rel="stylesheet" href="${AG}/styles/ag-grid.css">
<link rel="stylesheet" href="${AG}/styles/ag-theme-alpine.css">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--vscode-editor-background);
    color: var(--vscode-editor-foreground);
    font-family: var(--vscode-font-family, sans-serif);
    font-size: var(--vscode-font-size, 13px);
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }
  .pv-header {
    display: flex; align-items: center; gap: 4px;
    padding: 4px 8px;
    border-bottom: 1px solid var(--vscode-panel-border, #444);
    flex-shrink: 0;
  }
  .pv-nav { display: flex; align-items: center; gap: 4px; flex: 1; min-width: 0; }
  .pv-title {
    font-weight: 600; font-size: 12px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0;
  }
  .pv-counter { font-size: 11px; opacity: 0.6; white-space: nowrap; }
  .pv-btn {
    background: var(--vscode-button-secondaryBackground, transparent);
    color: var(--vscode-button-secondaryForeground, inherit);
    border: 1px solid var(--vscode-button-border, transparent);
    border-radius: 3px; cursor: pointer; padding: 2px 6px; font-size: 12px;
    line-height: 1.4;
  }
  .pv-btn:hover:not(:disabled) { background: var(--vscode-list-hoverBackground, rgba(255,255,255,0.1)); }
  .pv-btn:disabled { opacity: 0.35; cursor: default; }
  .pv-body { flex: 1; display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
  .pv-footer { flex-shrink: 0; padding: 3px 8px; font-size: 11px; opacity: 0.75;
    border-top: 1px solid var(--vscode-panel-border, #444); }
  .pv-footer-bar { display: flex; align-items: center; gap: 8px; }
  .pv-limit { width: 80px; padding: 1px 4px;
    background: var(--vscode-input-background); color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, #555); border-radius: 2px; font-size: 11px; }
  /* AG Grid zoom/fit toolbar */
  .pv-chart-toolbar {
    display: flex; align-items: center; gap: 4px; padding: 2px 8px;
    background: var(--vscode-editorWidget-background, var(--vscode-editor-background));
    border-bottom: 1px solid var(--vscode-panel-border, #444); flex-shrink: 0;
  }
  .pv-canvas-label { font-size: 11px; opacity: 0.7; margin-left: 8px; }
  /* AG Grid container */
  .pv-ag-container { height: 100%; width: 100%; }
  .pv-ag-idx { color: var(--vscode-editorLineNumber-foreground, #858585) !important;
    font-size: 10px !important; }
  .pv-ag-idx-header { background: transparent !important; }
  /* SelectPopupFilter */
  .pv-filter-list { max-height: 200px; overflow-y: auto; min-width: 120px; }
  .pv-filter-option {
    padding: 4px 8px; cursor: pointer; font-size: 12px;
    color: var(--vscode-dropdown-foreground, inherit);
  }
  .pv-filter-option:hover { background: var(--vscode-list-hoverBackground); }
  .pv-filter-option.pv-filter-selected { background: var(--vscode-list-activeSelectionBackground);
    color: var(--vscode-list-activeSelectionForeground); }
  /* Chart scroll/pan area */
  .pv-chart-scroll {
    flex: 1; overflow: auto; cursor: grab; min-height: 0;
    display: flex; align-items: flex-start; justify-content: flex-start;
  }
  .pv-chart-img { display: block; transform-origin: top left; }
  /* Page (canvas) view */
  .pv-page-view {
    flex: 1; overflow: auto; min-height: 0;
    display: flex; justify-content: center; padding: 16px;
    background: var(--vscode-editorWidget-background, var(--vscode-editor-background));
  }
  .pv-page {
    position: relative; flex-shrink: 0;
    background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }
  .pv-page-chart-img { position: absolute; }
  /* AG Grid */
  .ag-theme-alpine { --ag-font-size: 12px; }
  .ag-theme-alpine .ag-popup .ag-popup-child {
    background: #ffffff !important; border: 1px solid #d4d4d4; color: #1f1f1f !important;
  }
  .ag-theme-alpine .ag-popup .ag-popup-child * { color: inherit; }
  .ag-theme-alpine .ag-popup input.ag-input-field-input { padding-left: 24px !important; }
  .pv-empty { display: flex; align-items: center; justify-content: center;
    height: 100%; opacity: 0.4; font-size: 13px; }
  .pv-main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  /* Column pin context menu */
  .pv-pin-menu {
    position: fixed; z-index: 9999;
    background: var(--vscode-menu-background, #2d2d2d);
    border: 1px solid var(--vscode-menu-border, #555);
    border-radius: 3px; box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    padding: 2px 0; min-width: 140px; font-size: 12px;
  }
  .pv-pin-item {
    padding: 5px 12px; cursor: pointer;
    color: var(--vscode-menu-foreground, inherit); white-space: nowrap;
  }
  .pv-pin-item:hover { background: var(--vscode-menu-selectionBackground, #094771); }
  /* Column flash animation */
  @keyframes pv-col-flash-anim {
    0%   { background: rgba(0, 120, 212, 0.4); }
    100% { background: transparent; }
  }
  .pv-col-flash { animation: pv-col-flash-anim 0.8s ease-out forwards; }
</style>
</head>
<body>
<div class="pv-header">
  <div class="pv-nav">
    <button class="pv-btn" id="btn-back"  title="Back">&#9664;</button>
    <button class="pv-btn" id="btn-fwd"   title="Forward">&#9654;</button>
    <span class="pv-title" id="pv-title">—</span>
    <span class="pv-counter" id="pv-counter"></span>
  </div>
  <button class="pv-btn" id="btn-copy"    title="Copy to clipboard">&#128203;</button>
  <button class="pv-btn" id="btn-refresh" title="Refresh">&#8635;</button>
  <button class="pv-btn" id="btn-del"     title="Delete object">&#10005;</button>
  <button class="pv-btn" id="btn-clear"   title="Delete all">&#128465;</button>
</div>
<div class="pv-body" id="pv-body">
  <div class="pv-main" id="pv-main"><div class="pv-empty">No data yet — run a Pivotal cell</div></div>
</div>
<div class="pv-footer" id="pv-footer"></div>

<script nonce="${nonce}" src="${AG}/dist/ag-grid-community.min.js"></script>
<script nonce="${nonce}">
(function () {
  'use strict';

  const vscodeApi = acquireVsCodeApi();
  const { createGrid, AllCommunityModule, ModuleRegistry } = agGrid;
  ModuleRegistry.registerModules([AllCommunityModule]);

  const DEFAULT_LIMIT = 20000;
  const BASE_ROW_H = 26;
  const BASE_HDR_H = 30;

  // ── state ──────────────────────────────────────────────────────────────────
  const _latest = new Map();   // name → payload
  const _names  = [];          // ordered list of names
  let   _index  = -1;
  const _dfCache = new Map();  // name → { body, footer, api, applyZoom }
  let   _zoomCb  = null;
  let   _panelResizeCb = null;

  // ── DOM refs ───────────────────────────────────────────────────────────────
  const titleEl   = document.getElementById('pv-title');
  const counterEl = document.getElementById('pv-counter');
  const backBtn   = document.getElementById('btn-back');
  const fwdBtn    = document.getElementById('btn-fwd');
  const copyBtn   = document.getElementById('btn-copy');
  const refreshBtn= document.getElementById('btn-refresh');
  const delBtn    = document.getElementById('btn-del');
  const clearBtn  = document.getElementById('btn-clear');
  const bodyEl    = document.getElementById('pv-main');
  const footerEl  = document.getElementById('pv-footer');
  let   _pinMenu  = null;

  backBtn.disabled = fwdBtn.disabled = copyBtn.disabled =
    refreshBtn.disabled = delBtn.disabled = clearBtn.disabled = true;

  // ── SelectPopupFilter (AG Grid custom filter for categorical/boolean) ──────
  class SelectPopupFilter {
    init(params) {
      this._params = params;
      this._value  = '';
      const values = params.values ?? [];
      this._list = document.createElement('div');
      this._list.className = 'pv-filter-list';
      for (const v of values) {
        const item = document.createElement('div');
        item.className = 'pv-filter-option';
        item.dataset.value = v;
        item.textContent = v === '' ? 'All' : String(v);
        if (v === '') item.classList.add('pv-filter-selected');
        item.addEventListener('click', () => {
          this._value = v;
          this._list.querySelectorAll('.pv-filter-option').forEach(el =>
            el.classList.toggle('pv-filter-selected', el.dataset.value === v)
          );
          params.filterChangedCallback();
        });
        this._list.appendChild(item);
      }
      this._gui = document.createElement('div');
      this._gui.appendChild(this._list);
    }
    getGui()  { return this._gui; }
    doesFilterPass(params) {
      if (!this.isFilterActive()) return true;
      const colId = this._params.column.getColId();
      return String((params.data)[colId] ?? '') === this._value;
    }
    isFilterActive() { return this._value !== ''; }
    getModel() {
      return this.isFilterActive() ? { filterType: 'text', filter: this._value } : null;
    }
    setModel(model) {
      this._value = model?.filter ?? '';
      this._list?.querySelectorAll('.pv-filter-option').forEach(el =>
        el.classList.toggle('pv-filter-selected', el.dataset.value === this._value)
      );
    }
    destroy() {}
  }

  // ── navigation ─────────────────────────────────────────────────────────────
  function push(msg) {
    const isNew = !_latest.has(msg.name);
    _latest.set(msg.name, msg);
    if (!isNew) {
      _dfCache.get(msg.name)?.api?.destroy();
      _dfCache.delete(msg.name);
    }
    if (isNew) _names.push(msg.name);
    if (msg.hidden) {
      // Update data silently — re-render only if this item is already on screen
      if (_names.indexOf(msg.name) === _index) render();
    } else {
      _index = _names.indexOf(msg.name);
      render();
    }
  }

  function back()    { if (_index > 0) { _index--; render(); } }
  function forward() { if (_index < _names.length - 1) { _index++; render(); } }

  function focusItem(name) {
    const idx = _names.indexOf(name);
    if (idx >= 0 && idx !== _index) { _index = idx; render(); }
  }

  function deleteItemLocal(name) {
    const idx = _names.indexOf(name);
    if (idx < 0) return;
    _latest.delete(name);
    _dfCache.get(name)?.api?.destroy();
    _dfCache.delete(name);
    _names.splice(idx, 1);
    if (_names.length === 0) {
      clearLocal();
    } else {
      _index = Math.min(Math.max(_index, 0), _names.length - 1);
      render();
    }
  }

  function deleteCurrent() {
    if (_index < 0 || !_names.length) return;
    const name = _names[_index];
    vscodeApi.postMessage({ type: 'delete', name });
    deleteItemLocal(name);
  }

  function clearLocal() {
    _latest.clear();
    _dfCache.forEach(e => e.api?.destroy());
    _dfCache.clear();
    _names.length = 0;
    _index = -1;
    titleEl.textContent = '—';
    counterEl.textContent = '';
    backBtn.disabled = fwdBtn.disabled = copyBtn.disabled =
      refreshBtn.disabled = delBtn.disabled = clearBtn.disabled = true;
    bodyEl.innerHTML = '<div class="pv-empty">No data yet — run a Pivotal cell</div>';
    footerEl.innerHTML = '';
    _zoomCb = null; _panelResizeCb = null;
  }

  function clearAll() {
    vscodeApi.postMessage({ type: 'clear' });
    clearLocal();
  }

  function refreshCurrent() {
    if (_index < 0 || !_names.length) return;
    const name = _names[_index];
    const inp = footerEl.querySelector('.pv-limit');
    const limit = inp ? Math.max(100, parseInt(inp.value, 10) || DEFAULT_LIMIT) : DEFAULT_LIMIT;
    vscodeApi.postMessage({ type: 'request', name, limit });
  }

  // ── top-level render ───────────────────────────────────────────────────────
  function render() {
    if (_index < 0 || !_names.length) return;
    const p = _latest.get(_names[_index]);
    if (!p) return;
    _panelResizeCb = null;
    _zoomCb = null;

    const typeLabel = p.type === 'dataframe' ? 'DataFrame' : p.type === 'chart' ? 'Chart' : 'Table';
    titleEl.textContent = p.name + ' · ' + typeLabel;
    counterEl.textContent = (_index + 1) + ' / ' + _names.length;
    backBtn.disabled    = _index === 0;
    fwdBtn.disabled     = _index === _names.length - 1;
    copyBtn.disabled    = false;
    delBtn.disabled     = false;
    clearBtn.disabled   = false;
    refreshBtn.disabled = p.type !== 'dataframe';

    while (bodyEl.firstChild) bodyEl.removeChild(bodyEl.firstChild);
    while (footerEl.firstChild) footerEl.removeChild(footerEl.firstChild);

    if (p.type === 'dataframe') {
      renderDataFrame(p);
    } else {
      if (p.type === 'chart') renderChart(p);
      else                    renderGtTable(p);
    }
  }

  // ── DataFrame (AG Grid) ────────────────────────────────────────────────────
  function renderDataFrame(p) {
    const cached = _dfCache.get(p.name);
    if (cached) {
      bodyEl.appendChild(cached.body);
      footerEl.appendChild(cached.footer);
      _zoomCb = cached.applyZoom;
      return;
    }

    const { columns, data, dtypes } = p;
    const sigFigs = p.viewer_num_format ?? 5;
    const floatFmt = sigFigs > 0
      ? params => {
          const v = params.value;
          if (v === null || v === undefined || v === '') return '';
          const n = Number(v);
          if (isNaN(n)) return String(v);
          const s = n.toPrecision(sigFigs);
          return s.includes('e') ? s : String(parseFloat(s));
        }
      : undefined;

    const rowData = data.map((row, i) => {
      const obj = { _idx: i };
      columns.forEach((col, ci) => { obj[col] = row[ci]; });
      return obj;
    });

    const colDefs = [
      {
        field: '_idx', headerName: '', pinned: 'left',
        width: 52, minWidth: 52, maxWidth: 52,
        sortable: false, filter: false, floatingFilter: false,
        resizable: false, suppressMovable: true,
        cellClass: 'pv-ag-idx', headerClass: 'pv-ag-idx-header',
      },
      ...columns.map(col => {
        const dt = dtypes[col] ?? '';
        const isFloat = dt.startsWith('float');
        const isNum   = isFloat || dt.startsWith('int');
        const semType = (p.col_types ?? {})[col] ?? (isNum ? 'numeric' : 'string');
        const def = {
          field: col, headerName: col,
          type: isNum ? 'numericColumn' : undefined,
          headerTooltip: dt || undefined,
          resizable: true, sortable: true,
        };
        if (semType === 'categorical' || semType === 'boolean') {
          const vals = [...new Set(rowData.map(r => String(r[col] ?? '')))]
            .filter(v => v !== '').sort((a, b) => a.localeCompare(b));
          def.filter = SelectPopupFilter;
          def.filterParams = { values: ['', ...vals] };
        } else if (semType === 'numeric') {
          def.filter = 'agNumberColumnFilter';
        } else {
          def.filter = 'agTextColumnFilter';
          def.filterParams = { filterOptions: ['contains'], defaultOption: 'contains' };
        }
        if (isFloat && floatFmt) def.valueFormatter = floatFmt;
        return def;
      }),
    ];

    let zoomFactor = p.viewer_font ?? 1.0;
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;flex-direction:column;flex:1;min-height:0;height:100%;';

    const zoomToolbar = document.createElement('div');
    zoomToolbar.className = 'pv-chart-toolbar';
    zoomToolbar.innerHTML =
      '<button class="pv-btn pv-zoom-in"    title="Zoom in (+)">+</button>' +
      '<button class="pv-btn pv-zoom-out"   title="Zoom out (-)">&#8722;</button>' +
      '<button class="pv-btn pv-zoom-reset" title="Reset zoom">1:1</button>' +
      '<button class="pv-btn pv-fit-cols"   title="Fit columns">&#8596;</button>';

    const viewport = document.createElement('div');
    viewport.style.cssText = 'position:relative;overflow:hidden;flex:1;min-height:0;';

    const container = document.createElement('div');
    container.className = 'pv-ag-container ag-theme-alpine';
    container.style.cssText = 'position:absolute;top:0;left:0;transform-origin:top left;';
    container.style.width  = (100 / zoomFactor) + '%';
    container.style.height = (100 / zoomFactor) + '%';
    container.style.transform = 'scale(' + zoomFactor + ')';

    viewport.appendChild(container);
    wrapper.appendChild(zoomToolbar);
    wrapper.appendChild(viewport);
    bodyEl.appendChild(wrapper);  // append before createGrid

    const api = createGrid(container, {
      rowData, columnDefs: colDefs,
      rowHeight: BASE_ROW_H, headerHeight: BASE_HDR_H,
      defaultColDef: { sortable: true, filter: true, resizable: true, minWidth: 60, maxWidth: 300, width: 120 },
      suppressFieldDotNotation: true,
      suppressHeaderMenuButton: true,   // hide ≡ menu icon — doesn't respect :hover in VS Code webviews
      animateRows: false,
      enableCellTextSelection: true,
      localeText: { filterOoo: '' },
      onFirstDataRendered: e => {
        setTimeout(() => { if (container.clientWidth > 0) e.api.autoSizeAllColumns(false); }, 600);
      },
    });

    // Column pin: right-click a header cell to toggle pinned-left
    const hidePinMenu = () => { if (_pinMenu) { _pinMenu.remove(); _pinMenu = null; } };
    container.addEventListener('contextmenu', e => {
      const headerCell = e.target.closest('.ag-header-cell');
      if (!headerCell) return;
      const colId = headerCell.getAttribute('col-id');
      if (!colId || colId === '_idx') return;
      e.preventDefault();
      hidePinMenu();
      const isPinned = api.getColumnState().find(s => s.colId === colId)?.pinned === 'left';
      const menu = document.createElement('div');
      menu.className = 'pv-pin-menu';
      const item = document.createElement('div');
      item.className = 'pv-pin-item';
      item.textContent = isPinned ? '📌 Unpin column' : '📌 Pin to left';
      item.addEventListener('click', () => {
        api.applyColumnState({ state: [{ colId, pinned: isPinned ? null : 'left' }] });
        hidePinMenu();
      });
      menu.appendChild(item);
      menu.style.left = e.clientX + 'px';
      menu.style.top  = e.clientY + 'px';
      document.body.appendChild(menu);
      _pinMenu = menu;
      setTimeout(() => document.addEventListener('click', hidePinMenu, { once: true }), 0);
    });

    
    const applyZoom = mult => {
      zoomFactor = Math.max(0.5, Math.min(3.0, zoomFactor * mult));
      container.style.width  = (100 / zoomFactor) + '%';
      container.style.height = (100 / zoomFactor) + '%';
      container.style.transform = 'scale(' + zoomFactor + ')';
    };
    zoomToolbar.querySelector('.pv-zoom-in')   .addEventListener('click', () => applyZoom(1.25));
    zoomToolbar.querySelector('.pv-zoom-out')  .addEventListener('click', () => applyZoom(1 / 1.25));
    zoomToolbar.querySelector('.pv-zoom-reset').addEventListener('click', () => {
      zoomFactor = 1.0;
      container.style.width = container.style.height = '100%';
      container.style.transform = 'scale(1)';
    });
    zoomToolbar.querySelector('.pv-fit-cols')  .addEventListener('click', () => api.autoSizeAllColumns(false));
    _zoomCb = applyZoom;

    // Footer
    const [totalRows, totalCols] = p.shape;
    const truncMsg = p.truncated
      ? 'Showing ' + data.length.toLocaleString() + ' of ' + totalRows.toLocaleString() + ' rows'
      : totalRows.toLocaleString() + ' rows \\xD7 ' + totalCols + ' cols';
    const footer = document.createElement('div');
    footer.className = 'pv-footer-bar';
    footer.innerHTML = '<span class="pv-shape">' + truncMsg + '</span>' +
      (p.truncated
        ? '<label class="pv-limit-label">Show: <input class="pv-limit" type="number" value="' +
          data.length + '" min="100" step="1000"> rows</label>' +
          '<button class="pv-btn pv-show-all" title="Load all rows">Show all</button>'
        : '');
    if (p.truncated) {
      const inp = footer.querySelector('.pv-limit');
      inp.addEventListener('change', () => {
        const limit = Math.max(100, parseInt(inp.value, 10) || DEFAULT_LIMIT);
        vscodeApi.postMessage({ type: 'request', name: p.name, limit });
      });
      footer.querySelector('.pv-show-all').addEventListener('click', () => {
        vscodeApi.postMessage({ type: 'request', name: p.name, limit: p.shape[0] });
      });
    }
    _dfCache.set(p.name, { body: wrapper, footer, api, applyZoom });
    footerEl.appendChild(footer);
  }

  // ── Chart ──────────────────────────────────────────────────────────────────
  function renderChart(p) {
    if (p.canvas) renderChartOnPage(p); else renderChartFree(p);
  }

  function renderChartFree(p) {
    let scale = 1.0;
    let dragging = false, dragStartX = 0, dragStartY = 0, scrollStartX = 0, scrollStartY = 0;

    const toolbar = document.createElement('div');
    toolbar.className = 'pv-chart-toolbar';
    toolbar.innerHTML =
      '<button class="pv-btn pv-zoom-in"    title="Zoom in">+</button>' +
      '<button class="pv-btn pv-zoom-out"   title="Zoom out">&#8722;</button>' +
      '<button class="pv-btn pv-zoom-reset" title="Reset zoom">1:1</button>';

    const scroll = document.createElement('div');
    scroll.className = 'pv-chart-scroll';

    const img = document.createElement('img');
    img.className = 'pv-chart-img';
    img.src = 'data:image/png;base64,' + p.data;
    img.draggable = false;
    scroll.appendChild(img);

    bodyEl.appendChild(toolbar);
    bodyEl.appendChild(scroll);

    const setScale = s => {
      scale = Math.max(0.2, Math.min(5, s));
      img.style.transform = 'scale(' + scale + ')';
      img.style.transformOrigin = 'top left';
    };
    _zoomCb = f => setScale(scale * f);

    toolbar.querySelector('.pv-zoom-in')   .addEventListener('click', () => setScale(scale * 1.25));
    toolbar.querySelector('.pv-zoom-out')  .addEventListener('click', () => setScale(scale / 1.25));
    toolbar.querySelector('.pv-zoom-reset').addEventListener('click', () => setScale(1));

    scroll.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      dragging = true; dragStartX = e.clientX; dragStartY = e.clientY;
      scrollStartX = scroll.scrollLeft; scrollStartY = scroll.scrollTop;
      scroll.setPointerCapture(e.pointerId); scroll.style.cursor = 'grabbing';
    });
    scroll.addEventListener('pointermove', e => {
      if (!dragging) return;
      scroll.scrollLeft = scrollStartX - (e.clientX - dragStartX);
      scroll.scrollTop  = scrollStartY - (e.clientY - dragStartY);
    });
    scroll.addEventListener('pointerup', () => { dragging = false; scroll.style.cursor = 'grab'; });
  }

  function renderChartOnPage(p) {
    const cm = p.canvas;
    let userScale = 1.0;

    const toolbar = document.createElement('div');
    toolbar.className = 'pv-chart-toolbar';
    toolbar.innerHTML =
      '<button class="pv-btn pv-zoom-in"    title="Zoom in">+</button>' +
      '<button class="pv-btn pv-zoom-out"   title="Zoom out">&#8722;</button>' +
      '<button class="pv-btn pv-zoom-reset" title="Fit to panel">Fit</button>' +
      '<span class="pv-canvas-label">' + cm.label + ' \\xB7 ' + cm.margin_mm + 'mm margins</span>';

    const outer = document.createElement('div');
    outer.className = 'pv-page-view';

    const page = document.createElement('div');
    page.className = 'pv-page';

    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + p.data;
    img.className = 'pv-page-chart-img';
    img.draggable = false;

    page.appendChild(img);
    outer.appendChild(page);
    bodyEl.appendChild(toolbar);
    bodyEl.appendChild(outer);

    let lastAvailW = -1, rafId = 0;
    const apply = () => {
      const availW = Math.max(outer.clientWidth - 64, 80);
      if (Math.abs(availW - lastAvailW) < 1 && rafId === 0) return;
      lastAvailW = availW; rafId = 0;
      const pxPerMm = (availW / cm.page_width_mm) * userScale;
      page.style.width  = (cm.page_width_mm  * pxPerMm) + 'px';
      page.style.height = (cm.page_height_mm * pxPerMm) + 'px';
      img.style.width   = ((cm.chart_width_mm  ?? cm.page_width_mm  - 2 * cm.margin_mm) * pxPerMm) + 'px';
      img.style.height  = ((cm.chart_height_mm ?? cm.page_height_mm - 2 * cm.margin_mm) * pxPerMm) + 'px';
      img.style.left    = (cm.margin_mm * pxPerMm) + 'px';
      img.style.top     = (cm.margin_mm * pxPerMm) + 'px';
    };
    const applyForced = () => { lastAvailW = -1; apply(); };
    _panelResizeCb = applyForced;
    _zoomCb = f => { userScale *= f; applyForced(); };

    toolbar.querySelector('.pv-zoom-in')   .addEventListener('click', () => { userScale *= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-out')  .addEventListener('click', () => { userScale /= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-reset').addEventListener('click', () => { userScale = 1.0;   applyForced(); });

    new ResizeObserver(() => { cancelAnimationFrame(rafId); rafId = requestAnimationFrame(apply); }).observe(outer);
    requestAnimationFrame(apply);
  }

  // ── GT Table ───────────────────────────────────────────────────────────────
  function renderGtTable(p) {
    if (p.canvas) renderGtTableOnPage(p); else renderGtTableFree(p);
  }

  function renderGtTableFree(p) {
    const iframe = document.createElement('iframe');
    iframe.srcdoc = p.html;
    iframe.setAttribute('sandbox', 'allow-same-origin allow-scripts');
    iframe.style.cssText = 'flex:1;width:100%;height:100%;border:none;';
    bodyEl.appendChild(iframe);
  }

  function renderGtTableOnPage(p) {
    const cm = p.canvas;
    let userScale = 1.0;
    const usableW_mm = cm.page_width_mm - 2 * cm.margin_mm;
    const CSS_PX_PER_MM = 96 / 25.4;
    const loadW = Math.round(usableW_mm * CSS_PX_PER_MM);

    const toolbar = document.createElement('div');
    toolbar.className = 'pv-chart-toolbar';
    toolbar.innerHTML =
      '<button class="pv-btn pv-zoom-in"    title="Zoom in">+</button>' +
      '<button class="pv-btn pv-zoom-out"   title="Zoom out">&#8722;</button>' +
      '<button class="pv-btn pv-zoom-reset" title="Fit to panel">Fit</button>' +
      '<span class="pv-canvas-label">' + cm.label + ' \\xB7 ' + cm.margin_mm + 'mm margins</span>';

    const outer = document.createElement('div');
    outer.className = 'pv-page-view';

    const page = document.createElement('div');
    page.className = 'pv-page';

    const iframe = document.createElement('iframe');
    iframe.srcdoc = p.html;
    iframe.setAttribute('sandbox', 'allow-same-origin allow-scripts');
    iframe.style.cssText = 'position:absolute;border:none;visibility:hidden;width:' + loadW + 'px;';

    page.appendChild(iframe);
    outer.appendChild(page);
    bodyEl.appendChild(toolbar);
    bodyEl.appendChild(outer);

    let naturalW = 0, naturalH = 0, lastAvailW = -1, rafId = 0;
    const apply = () => {
      if (naturalW === 0) return;
      const availW = Math.max(outer.clientWidth - 64, 80);
      if (Math.abs(availW - lastAvailW) < 1 && rafId === 0) return;
      lastAvailW = availW; rafId = 0;
      const pxPerMm  = (availW / cm.page_width_mm) * userScale;
      const marginPx = cm.margin_mm * pxPerMm;
      const naturalW_mm = naturalW / CSS_PX_PER_MM;
      const targetW_mm  = Math.min(naturalW_mm, usableW_mm);
      const scale = (targetW_mm * pxPerMm) / naturalW;
      page.style.width  = (cm.page_width_mm  * pxPerMm) + 'px';
      page.style.height = (cm.page_height_mm * pxPerMm) + 'px';
      iframe.style.left            = marginPx + 'px';
      iframe.style.top             = marginPx + 'px';
      iframe.style.transform       = 'scale(' + scale + ')';
      iframe.style.transformOrigin = '0 0';
    };
    const applyForced = () => { lastAvailW = -1; apply(); };
    _panelResizeCb = applyForced;
    _zoomCb = f => { userScale *= f; applyForced(); };

    iframe.addEventListener('load', () => {
      try {
        const doc = iframe.contentDocument;
        naturalW = Math.max(doc.documentElement.scrollWidth, doc.body?.scrollWidth ?? 0, 1);
        naturalH = Math.max(doc.documentElement.scrollHeight, doc.body?.scrollHeight ?? 0, 1);
      } catch (_) { naturalW = 800; naturalH = 600; }
      iframe.style.width      = naturalW + 'px';
      iframe.style.height     = naturalH + 'px';
      iframe.style.visibility = '';
      applyForced();
    });

    toolbar.querySelector('.pv-zoom-in')   .addEventListener('click', () => { userScale *= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-out')  .addEventListener('click', () => { userScale /= 1.25; applyForced(); });
    toolbar.querySelector('.pv-zoom-reset').addEventListener('click', () => { userScale = 1.0;   applyForced(); });

    new ResizeObserver(() => { cancelAnimationFrame(rafId); rafId = requestAnimationFrame(apply); }).observe(outer);
  }

  // ── Clipboard ──────────────────────────────────────────────────────────────
  async function copyToClipboard() {
    if (_index < 0 || !_names.length) return;
    const p = _latest.get(_names[_index]);
    if (!p) return;
    try {
      if (p.type === 'dataframe') {
        const tsv = [p.columns.join('\\t'),
          ...p.data.map(row => row.map(v => (v == null ? '' : String(v))).join('\\t'))].join('\\n');
        const html = '<table><thead><tr>' + p.columns.map(c => '<th>' + c + '</th>').join('') +
          '</tr></thead><tbody>' +
          p.data.map(row => '<tr>' + row.map(v => '<td>' + (v == null ? '' : String(v)) + '</td>').join('') + '</tr>').join('') +
          '</tbody></table>';
        await navigator.clipboard.write([new ClipboardItem({
          'text/plain': new Blob([tsv],  { type: 'text/plain' }),
          'text/html':  new Blob([html], { type: 'text/html'  }),
        })]);
      } else if (p.type === 'chart') {
        const blob = await fetch('data:image/png;base64,' + p.data).then(r => r.blob());
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      } else {
        const html = p.html;
        await navigator.clipboard.write([new ClipboardItem({
          'text/html':  new Blob([html], { type: 'text/html'  }),
          'text/plain': new Blob([html], { type: 'text/plain' }),
        })]);
      }
      const orig = copyBtn.textContent;
      copyBtn.textContent = '\\u2713';
      setTimeout(() => { copyBtn.textContent = orig; }, 1200);
    } catch (err) {
      console.error('[Pivotal] clipboard copy failed:', err);
    }
  }

  // ── Keyboard shortcuts ─────────────────────────────────────────────────────
  document.addEventListener('keydown', e => {
    if (e.altKey && e.key === '[') { back(); e.preventDefault(); }
    if (e.altKey && e.key === ']') { forward(); e.preventDefault(); }
  });

  // ── Button wiring ──────────────────────────────────────────────────────────
  backBtn.addEventListener('click', back);
  fwdBtn.addEventListener('click', forward);
  copyBtn.addEventListener('click', copyToClipboard);
  refreshBtn.addEventListener('click', refreshCurrent);
  delBtn.addEventListener('click', deleteCurrent);
  clearBtn.addEventListener('click', clearAll);

  // ── ResizeObserver for panel resize callbacks ──────────────────────────────
  new ResizeObserver(() => { _panelResizeCb?.(); }).observe(bodyEl);

  // ── Message handler (from extension host → webview) ────────────────────────
  window.addEventListener('message', event => {
    const msg = event.data;
    if (!msg || !msg.type) return;
    switch (msg.type) {
      case 'dataframe':
      case 'chart':
      case 'gt_table':
        push(msg);
        break;
      case 'delete':
        deleteItemLocal(msg.name);
        break;
      case 'clear':
        clearLocal();
        break;
      case 'focus':
        focusItem(msg.name);
        break;
      case 'scroll_col': {
        // Scroll the current AG Grid to the named column and flash it
        const cached = _dfCache.get(msg.name);
        if (cached) {
          cached.api.ensureColumnVisible(msg.col);
          const escaped = msg.col.replace(/"/g, '\\"');
          const hdr = cached.body.querySelector('.ag-header-cell[col-id="' + escaped + '"]');
          if (hdr) { hdr.classList.add('pv-col-flash'); setTimeout(() => hdr.classList.remove('pv-col-flash'), 800); }
        }
        break;
      }
    }
  });

  // Signal to the extension that the webview is ready to receive messages.
  // The extension will replay any items that arrived while the webview was loading.
  vscodeApi.postMessage({ type: 'ready' });

})();
</script>
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Phase 4 — TreeView Explorer
// ---------------------------------------------------------------------------

type ExplorerNode =
  | { kind: 'category'; label: string; itemType: string }
  | { kind: 'item';     type: string; name: string; payload: Record<string, unknown> }
  | { kind: 'column';   parent: string; col: string; dtype: string; semType: string };

/** Shared store: name → full bridge payload, updated on every bridge message. */
const _explorerItems = new Map<string, Record<string, unknown>>();

let _explorerProvider: _ExplorerProvider | null = null;
let _pivotalTreeView: vscode.TreeView<ExplorerNode> | null = null;

function _refreshExplorer(_type?: string): void {
  _explorerProvider?.refresh();
}

const _CATEGORIES = [
  { label: 'Data',   itemType: 'dataframe' },
  { label: 'Charts', itemType: 'chart'     },
  { label: 'Tables', itemType: 'gt_table'  },
];

class _ExplorerProvider implements vscode.TreeDataProvider<ExplorerNode> {
  private readonly _emitter = new vscode.EventEmitter<ExplorerNode | undefined>();
  readonly onDidChangeTreeData = this._emitter.event;

  filter = '';   // current search string (lower-cased); empty = show all

  refresh(): void { this._emitter.fire(undefined); }

  getTreeItem(node: ExplorerNode): vscode.TreeItem {
    if (node.kind === 'category') {
      const item = new vscode.TreeItem(
        node.label,
        vscode.TreeItemCollapsibleState.Expanded,
      );
      item.contextValue = 'pivotalCategory';
      return item;
    }

    if (node.kind === 'column') {
      const iconMap: Record<string, string> = {
        numeric: 'symbol-numeric', categorical: 'symbol-enum',
        datetime: 'calendar',      boolean: 'symbol-boolean',
        string: 'symbol-string',
      };
      const item = new vscode.TreeItem(node.col, vscode.TreeItemCollapsibleState.None);
      item.description  = node.dtype;
      item.iconPath     = new vscode.ThemeIcon(iconMap[node.semType] ?? 'symbol-string');
      item.contextValue = 'pivotalColumn';
      item.command = { command: 'pivotal.explorer.scrollToColumn', title: 'Scroll to column', arguments: [node] };
      return item;
    }

    // kind === 'item'
    const p = node.payload;
    const hasColumns = node.type === 'dataframe'
      && Array.isArray(p.columns) && (p.columns as string[]).length > 0;

    const item = new vscode.TreeItem(
      node.name,
      hasColumns
        ? vscode.TreeItemCollapsibleState.Collapsed
        : vscode.TreeItemCollapsibleState.None,
    );

    if (node.type === 'dataframe') {
      const shape = p.shape as [number, number] | undefined;
      item.description  = shape ? `${shape[0].toLocaleString()} × ${shape[1]}` : '';
      item.iconPath     = new vscode.ThemeIcon('table');
      item.contextValue = 'pivotalDataframe';
    } else if (node.type === 'chart') {
      item.description  = 'chart';
      item.iconPath     = new vscode.ThemeIcon('graph');
      item.contextValue = 'pivotalChart';
    } else {
      item.description  = 'table';
      item.iconPath     = new vscode.ThemeIcon('list-flat');
      item.contextValue = 'pivotalTable';
    }

    item.command = { command: 'pivotal.explorer.view', title: 'View', arguments: [node] };
    return item;
  }

  getChildren(node?: ExplorerNode): ExplorerNode[] {
    const f = this.filter;
    if (!node) {
      // Root: show only categories that have at least one (matching) item
      return _CATEGORIES
        .filter(c => [..._explorerItems.entries()].some(
          ([name, p]) => p.type === c.itemType && (!f || name.toLowerCase().includes(f))
        ))
        .map(c => ({ kind: 'category' as const, label: c.label, itemType: c.itemType }));
    }
    if (node.kind === 'category') {
      const result: ExplorerNode[] = [];
      for (const [name, payload] of _explorerItems) {
        if ((payload.type as string) === node.itemType && (!f || name.toLowerCase().includes(f))) {
          result.push({ kind: 'item', type: node.itemType, name, payload });
        }
      }
      return result;
    }
    if (node.kind === 'item' && node.type === 'dataframe') {
      const columns  = (node.payload.columns   as string[])               ?? [];
      const dtypes   = (node.payload.dtypes    as Record<string, string>) ?? {};
      const colTypes = (node.payload.col_types as Record<string, string>) ?? {};
      return columns.map(col => ({
        kind:    'column' as const,
        parent:  node.name,
        col,
        dtype:   dtypes[col]   ?? '',
        semType: colTypes[col] ?? 'string',
      }));
    }
    return [];
  }
}

// ---------------------------------------------------------------------------
// Phase 5 — GUI Dialogs (Plot & Pivot WebviewPanels; Load/Save/Export QuickPick)
// ---------------------------------------------------------------------------

let _plotGuiPanel:  vscode.WebviewPanel | null = null;
let _pivotGuiPanel: vscode.WebviewPanel | null = null;

/** Last focused .pivotal text editor — kept even when a webview steals focus. */
let _lastPivotalEditor: vscode.TextEditor | undefined;

/** Collect dataframe info from the live explorer store and push to a GUI panel. */
function _sendTablesToGui(panel: vscode.WebviewPanel): void {
  const tables: Record<string, { columns: string[]; dtypes: Record<string, string> }> = {};
  for (const [name, payload] of _explorerItems) {
    if ((payload.type as string) === 'dataframe') {
      tables[name] = {
        columns: (payload.columns as string[]) ?? [],
        dtypes:  (payload.dtypes  as Record<string, string>) ?? {},
      };
    }
  }
  panel.webview.postMessage({ type: 'tables', tables });
}

/** Push updated table list to any open GUI panels (called after bridge updates). */
function _refreshGuiPanels(): void {
  if (_plotGuiPanel)  { _sendTablesToGui(_plotGuiPanel); }
  if (_pivotGuiPanel) { _sendTablesToGui(_pivotGuiPanel); }
}

function _buildPlotGuiHtml(): string {
  const nonce = _generateNonce();
  const csp = [
    `default-src 'none'`,
    `script-src 'nonce-${nonce}'`,
    `style-src 'unsafe-inline'`,
  ].join('; ');
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0; padding: 14px 16px 16px;
    background: var(--vscode-editor-background);
    color: var(--vscode-editor-foreground);
    font-family: var(--vscode-font-family, sans-serif);
    font-size: var(--vscode-font-size, 13px);
  }
  h2 { font-size: 13px; font-weight: 600; margin: 0 0 14px;
    border-bottom: 1px solid var(--vscode-panel-border, #444); padding-bottom: 8px; }
  .field { margin-bottom: 9px; }
  .drow { display: flex; gap: 8px; }
  .drow .field { flex: 1; min-width: 0; }
  label { display: block; font-size: 11px; opacity: 0.7; margin-bottom: 3px;
    text-transform: uppercase; letter-spacing: 0.04em; }
  select, input[type="text"] {
    width: 100%; padding: 3px 6px;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, #555);
    border-radius: 2px; font-size: 12px; font-family: inherit;
  }
  select:focus, input:focus { outline: 1px solid var(--vscode-focusBorder); outline-offset: -1px; }
  /* dynamic multi-col rows */
  .dyn-section { margin-bottom: 9px; }
  .dyn-section label { margin-bottom: 3px; }
  .col-row { display: flex; align-items: center; gap: 4px; margin-bottom: 3px; }
  .col-row select { flex: 1; }
  .rm-btn, .add-btn {
    flex-shrink: 0; width: 22px; height: 22px;
    background: var(--vscode-button-secondaryBackground, transparent);
    color: var(--vscode-button-secondaryForeground, inherit);
    border: 1px solid var(--vscode-button-border, #555);
    border-radius: 2px; cursor: pointer; font-size: 14px; line-height: 1;
    padding: 0; display: flex; align-items: center; justify-content: center;
  }
  .rm-btn:hover, .add-btn:hover { background: var(--vscode-list-hoverBackground); }
  .add-row { display: flex; gap: 4px; margin-top: 2px; }
  .add-btn.wide { width: auto; padding: 0 8px; font-size: 12px; }
  /* 3 action buttons */
  .btn-row { display: flex; gap: 6px; margin-top: 10px; }
  .act-btn {
    flex: 1; padding: 6px 4px;
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    border: none; border-radius: 2px;
    cursor: pointer; font-size: 11px; font-family: inherit; text-align: center;
  }
  .act-btn:hover { background: var(--vscode-button-hoverBackground); }
  .act-btn.secondary {
    background: var(--vscode-button-secondaryBackground, transparent);
    color: var(--vscode-button-secondaryForeground, inherit);
    border: 1px solid var(--vscode-button-border, #555);
  }
  .act-btn.secondary:hover { background: var(--vscode-list-hoverBackground); }
  .hint { font-size: 11px; opacity: 0.5; margin-top: 6px; }
</style>
</head>
<body>
<h2>Pivot Plot</h2>
<div class="drow">
  <div class="field" style="flex:1.2">
    <label>with (source)</label>
    <select id="table"></select>
  </div>
  <div class="field" style="flex:0.8">
    <label>Chart type</label>
    <select id="chartType">
      <option value="bar">bar</option>
      <option value="line">line</option>
      <option value="scatter">scatter</option>
      <option value="area">area</option>
    </select>
  </div>
</div>
<div class="field">
  <label>Chart name</label>
  <input type="text" id="chartName" value="temp">
</div>
<div class="field">
  <label>filter (optional)</label>
  <input type="text" id="filterExpr" placeholder="e.g. season > 2010">
</div>
<div class="drow">
  <div class="field">
    <label>x column</label>
    <select id="x"></select>
  </div>
  <div class="field">
    <label>x label (optional)</label>
    <input type="text" id="xLabel" placeholder="axis label">
  </div>
</div>
<div class="dyn-section">
  <label>y columns (agg func + column per row)</label>
  <div id="yRows"></div>
  <div class="add-row">
    <input type="text" id="yLabel" placeholder="y axis label (optional)" style="flex:1">
    <button class="add-btn wide" id="addYBtn">+ y column</button>
  </div>
</div>
<div class="field">
  <label>by (group-by, optional)</label>
  <select id="by"><option value="">— none —</option></select>
</div>
<div class="btn-row">
  <button class="act-btn" id="btnInsert">Insert at cursor</button>
  <button class="act-btn secondary" id="btnExecute">Execute</button>
</div>
<p class="hint">Place cursor in .pivotal file, then click Insert.</p>

<script nonce="${nonce}">
(function () {
  'use strict';
  const api = acquireVsCodeApi();
  let _tables = {};
  let _yCols = [];  // array of {sel} objects

  const tableEl      = document.getElementById('table');
  const chartTypeEl  = document.getElementById('chartType');
  const chartNameEl  = document.getElementById('chartName');
  const filterExprEl = document.getElementById('filterExpr');
  const xEl          = document.getElementById('x');
  const xLabelEl     = document.getElementById('xLabel');
  const yRowsEl      = document.getElementById('yRows');
  const yLabelEl     = document.getElementById('yLabel');
  const addYBtn      = document.getElementById('addYBtn');
  const byEl         = document.getElementById('by');

  const AGG_FUNCS = ['mean','sum','count','min','max','median','quantile','percentile'];

  function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

  function currentCols() {
    const tbl = tableEl.value;
    return (tbl && _tables[tbl]) ? _tables[tbl].columns : [];
  }

  function colOpts(cols, optional) {
    return (optional ? '<option value="">\\u2014 none \\u2014</option>' : '')
      + cols.map(c => '<option value="' + esc(c) + '">' + esc(c) + '</option>').join('');
  }

  function aggOpts(selected) {
    return AGG_FUNCS.map(f => '<option value="' + f + '"' + (f === selected ? ' selected' : '') + '>' + f + '</option>').join('');
  }

  function rebuildYRows() {
    yRowsEl.innerHTML = '';
    _yCols.forEach((item, i) => {
      const row = document.createElement('div');
      row.className = 'col-row';

      const aggSel = document.createElement('select');
      aggSel.style.cssText = 'width:72px;flex-shrink:0';
      aggSel.innerHTML = aggOpts(item.aggVal || 'mean');
      aggSel.addEventListener('change', () => { item.aggVal = aggSel.value; updatePreview(); });
      item.aggSel = aggSel;

      const colSel = document.createElement('select');
      colSel.innerHTML = colOpts(currentCols(), false);
      if (item.colVal) { colSel.value = item.colVal; }
      colSel.addEventListener('change', () => { item.colVal = colSel.value; updatePreview(); });
      item.colSel = colSel;

      const rm = document.createElement('button');
      rm.className = 'rm-btn'; rm.textContent = '\\u2212'; rm.title = 'Remove';
      rm.disabled = _yCols.length === 1;
      rm.addEventListener('click', () => { _yCols.splice(i, 1); rebuildYRows(); updatePreview(); });

      row.appendChild(aggSel); row.appendChild(colSel); row.appendChild(rm);
      yRowsEl.appendChild(row);
    });
  }

  function addYRow() {
    const cols = currentCols();
    _yCols.push({ aggVal: 'mean', colVal: cols[0] || '', aggSel: null, colSel: null });
    rebuildYRows();
    updatePreview();
  }

  function buildCode() {
    const tbl    = tableEl.value;
    const kind   = chartTypeEl.value;
    const name   = chartNameEl.value.trim();
    const filt   = filterExprEl.value.trim();
    const x      = xEl.value;
    const xl     = xLabelEl.value.trim();
    const yl     = yLabelEl.value.trim();
    const by     = byEl.value;
    const yEntries = _yCols
      .map(c => ({ agg: c.aggVal || (c.aggSel && c.aggSel.value) || 'mean', col: c.colVal || (c.colSel && c.colSel.value) || '' }))
      .filter(e => e.col);
    if (!tbl || !x || !yEntries.length) return '(select table, x and at least one y column)';
    const yStr = yEntries.map(e => e.agg + ' ' + e.col).join(', ');
    const lines = [
      'with ' + tbl,
      ...(filt ? ['filter ' + filt] : []),
      'pivot plot ' + kind + (name ? ' ' + name : ''),
      '    x ' + x + (xl ? ' "' + xl.replace(/"/g,'\\\\"') + '"' : ''),
      '    y ' + yStr + (yl ? ' "' + yl.replace(/"/g,'\\\\"') + '"' : ''),
    ];
    if (by) lines.push('    by ' + by);
    return lines.join('\\n');
  }

  function updatePreview() {}

  function populateCols() {
    const cols = currentCols();
    const prevX  = xEl.value;
    const prevBy = byEl.value;
    xEl.innerHTML  = colOpts(cols, false);
    byEl.innerHTML = colOpts(cols, true);
    if (prevX  && cols.indexOf(prevX)  !== -1) { xEl.value  = prevX; }
    if (prevBy && cols.indexOf(prevBy) !== -1) { byEl.value = prevBy; }
    _yCols.forEach(item => {
      if (item.colSel) { item.colSel.innerHTML = colOpts(cols, false); if (item.colVal) item.colSel.value = item.colVal; }
    });
    updatePreview();
  }

  function populateTables() {
    const names = Object.keys(_tables);
    const prevTable = tableEl.value;
    tableEl.innerHTML = names.length
      ? names.map(n => '<option value="' + esc(n) + '">' + esc(n) + '</option>').join('')
      : '<option value="">\\u2014 no data in session \\u2014</option>';
    if (prevTable && _tables[prevTable]) { tableEl.value = prevTable; }
    populateCols();
  }

  tableEl.addEventListener('change', populateCols);
  [chartTypeEl, xEl, byEl].forEach(el => el.addEventListener('change', updatePreview));
  [chartNameEl, filterExprEl, xLabelEl, yLabelEl].forEach(el => el.addEventListener('input', updatePreview));
  addYBtn.addEventListener('click', addYRow);

  function send(action) {
    const code = buildCode();
    if (code.startsWith('(')) return;
    api.postMessage({ action, code });
  }

  document.getElementById('btnInsert') .addEventListener('click', () => send('insert'));
  document.getElementById('btnExecute').addEventListener('click', () => send('execute'));

  window.addEventListener('message', event => {
    const msg = event.data;
    if (msg.type === 'tables') { _tables = msg.tables || {}; populateTables(); }
  });

  // Initialise with one Y row
  addYRow();
})();
</script>
</body>
</html>`;
}

function _buildPivotGuiHtml(): string {
  const nonce = _generateNonce();
  const csp = [
    `default-src 'none'`,
    `script-src 'nonce-${nonce}'`,
    `style-src 'unsafe-inline'`,
  ].join('; ');
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0; padding: 14px 16px 16px;
    background: var(--vscode-editor-background);
    color: var(--vscode-editor-foreground);
    font-family: var(--vscode-font-family, sans-serif);
    font-size: var(--vscode-font-size, 13px);
  }
  h2 { font-size: 13px; font-weight: 600; margin: 0 0 14px;
    border-bottom: 1px solid var(--vscode-panel-border, #444); padding-bottom: 8px; }
  .field { margin-bottom: 9px; }
  .drow { display: flex; gap: 8px; }
  .drow .field { flex: 1; min-width: 0; }
  label { display: block; font-size: 11px; opacity: 0.7; margin-bottom: 3px;
    text-transform: uppercase; letter-spacing: 0.04em; }
  select, input[type="text"] {
    width: 100%; padding: 3px 6px;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, #555);
    border-radius: 2px; font-size: 12px; font-family: inherit;
  }
  select:focus, input:focus { outline: 1px solid var(--vscode-focusBorder); outline-offset: -1px; }
  .dyn-section { margin-bottom: 9px; }
  .dyn-section label { margin-bottom: 3px; }
  .col-row { display: flex; align-items: center; gap: 4px; margin-bottom: 3px; }
  .col-row select { flex: 1; }
  .rm-btn, .add-btn {
    flex-shrink: 0; width: 22px; height: 22px;
    background: var(--vscode-button-secondaryBackground, transparent);
    color: var(--vscode-button-secondaryForeground, inherit);
    border: 1px solid var(--vscode-button-border, #555);
    border-radius: 2px; cursor: pointer; font-size: 14px; line-height: 1;
    padding: 0; display: flex; align-items: center; justify-content: center;
  }
  .rm-btn:hover, .add-btn:hover { background: var(--vscode-list-hoverBackground); }
  .add-btn.wide { width: auto; padding: 0 8px; font-size: 12px; }
  .btn-row { display: flex; gap: 6px; margin-top: 10px; }
  .act-btn {
    flex: 1; padding: 6px 4px;
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    border: none; border-radius: 2px;
    cursor: pointer; font-size: 11px; font-family: inherit; text-align: center;
  }
  .act-btn:hover { background: var(--vscode-button-hoverBackground); }
  .act-btn.secondary {
    background: var(--vscode-button-secondaryBackground, transparent);
    color: var(--vscode-button-secondaryForeground, inherit);
    border: 1px solid var(--vscode-button-border, #555);
  }
  .act-btn.secondary:hover { background: var(--vscode-list-hoverBackground); }
  .hint { font-size: 11px; opacity: 0.5; margin-top: 6px; }
</style>
</head>
<body>
<h2>Pivot Table</h2>
<div class="drow">
  <div class="field">
    <label>with (source)</label>
    <select id="table"></select>
  </div>
  <div class="field">
    <label>as (output)</label>
    <input type="text" id="alias" value="temp">
  </div>
</div>
<div class="dyn-section">
  <label>rows</label>
  <div id="rowsDiv"></div>
  <button class="add-btn wide" id="addRowBtn">+ row column</button>
</div>
<div class="dyn-section">
  <label>cols</label>
  <div id="colsDiv"></div>
  <button class="add-btn wide" id="addColBtn">+ col column</button>
</div>
<div class="drow">
  <div class="field">
    <label>agg function</label>
    <select id="aggFunc">
      <option value="mean">mean</option>
      <option value="sum">sum</option>
      <option value="count">count</option>
      <option value="min">min</option>
      <option value="max">max</option>
      <option value="median">median</option>
    </select>
  </div>
  <div class="field">
    <label>value column</label>
    <select id="valueCol"></select>
  </div>
</div>
<div class="btn-row">
  <button class="act-btn" id="btnInsert">Insert at cursor</button>
  <button class="act-btn secondary" id="btnExecute">Execute</button>
</div>
<p class="hint">Place cursor in .pivotal file, then click Insert.</p>

<script nonce="${nonce}">
(function () {
  'use strict';
  const api = acquireVsCodeApi();
  let _tables = {};
  let _rowCols = [];
  let _colCols = [];

  const tableEl   = document.getElementById('table');
  const aliasEl   = document.getElementById('alias');
  const rowsDiv   = document.getElementById('rowsDiv');
  const colsDiv   = document.getElementById('colsDiv');
  const aggEl     = document.getElementById('aggFunc');
  const valueEl   = document.getElementById('valueCol');

  function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

  function currentCols() {
    const tbl = tableEl.value;
    return (tbl && _tables[tbl]) ? _tables[tbl].columns : [];
  }

  function colOpts(cols) {
    return cols.map(c => '<option value="' + esc(c) + '">' + esc(c) + '</option>').join('');
  }

  function rebuildDynSection(items, container) {
    container.innerHTML = '';
    items.forEach((item, i) => {
      const row = document.createElement('div');
      row.className = 'col-row';
      const sel = document.createElement('select');
      sel.innerHTML = colOpts(currentCols());
      if (item.val) { sel.value = item.val; }
      sel.addEventListener('change', () => { item.val = sel.value; updatePreview(); });
      item.sel = sel;
      const rm = document.createElement('button');
      rm.className = 'rm-btn'; rm.textContent = '\\u2212'; rm.title = 'Remove';
      rm.disabled = items.length === 1;
      rm.addEventListener('click', () => { items.splice(i, 1); rebuildDynSection(items, container); updatePreview(); });
      row.appendChild(sel); row.appendChild(rm);
      container.appendChild(row);
    });
  }

  function addDynRow(items, container) {
    const cols = currentCols();
    items.push({ val: cols[0] || '', sel: null });
    rebuildDynSection(items, container);
    updatePreview();
  }

  function buildCode() {
    const tbl   = tableEl.value;
    const alias = aliasEl.value.trim();
    const func  = aggEl.value;
    const val   = valueEl.value;
    const rows  = _rowCols.map(c => c.val || (c.sel && c.sel.value) || '').filter(Boolean);
    const cols  = _colCols.map(c => c.val || (c.sel && c.sel.value) || '').filter(Boolean);
    if (!tbl || !rows.length || !cols.length || !val) {
      return '(select table, rows, cols and value column)';
    }
    const lines = [
      alias ? 'with ' + tbl + ' as ' + alias : 'with ' + tbl,
      'pivot',
      '    rows ' + rows.join(', '),
      '    cols ' + cols.join(', '),
      '    agg ' + func + '(' + val + ')',
    ];
    return lines.join('\\n');
  }

  function updatePreview() {}

  function populateCols() {
    const cols = currentCols();
    const prevVal = valueEl.value;
    valueEl.innerHTML = colOpts(cols);
    if (prevVal && cols.indexOf(prevVal) !== -1) { valueEl.value = prevVal; }
    _rowCols.forEach(item => { if (item.sel) { item.sel.innerHTML = colOpts(cols); if (item.val) item.sel.value = item.val; } });
    _colCols.forEach(item => { if (item.sel) { item.sel.innerHTML = colOpts(cols); if (item.val) item.sel.value = item.val; } });
    updatePreview();
  }

  function populateTables() {
    const names = Object.keys(_tables);
    const prevTable = tableEl.value;
    tableEl.innerHTML = names.length
      ? names.map(n => '<option value="' + esc(n) + '">' + esc(n) + '</option>').join('')
      : '<option value="">\\u2014 no data in session \\u2014</option>';
    if (prevTable && _tables[prevTable]) { tableEl.value = prevTable; }
    populateCols();
  }

  tableEl.addEventListener('change', populateCols);
  [aggEl, valueEl].forEach(el => el.addEventListener('change', updatePreview));
  aliasEl.addEventListener('input', updatePreview);
  document.getElementById('addRowBtn').addEventListener('click', () => addDynRow(_rowCols, rowsDiv));
  document.getElementById('addColBtn').addEventListener('click', () => addDynRow(_colCols, colsDiv));

  function send(action) {
    const code = buildCode();
    if (code.startsWith('(')) return;
    api.postMessage({ action, code });
  }

  document.getElementById('btnInsert') .addEventListener('click', () => send('insert'));
  document.getElementById('btnExecute').addEventListener('click', () => send('execute'));

  window.addEventListener('message', event => {
    const msg = event.data;
    if (msg.type === 'tables') { _tables = msg.tables || {}; populateTables(); }
  });

  // Initialise with one row and one col entry
  addDynRow(_rowCols, rowsDiv);
  addDynRow(_colCols, colsDiv);
})();
</script>
</body>
</html>`;
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
      _scheduleKernelCheck();
      _scheduleRapidBridgePoll();
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
        _scheduleKernelCheck();
        _scheduleRapidBridgePoll();
      } catch {
        vscode.window.showErrorMessage(
          'Pivotal: Failed to send selection to Interactive Window. ' +
          'Please ensure a Python kernel is running.'
        );
      }
    }
  );

  // --- Helper: resolve the active .pivotal file path ---
  // Uses _lastPivotalEditor first (works even when focus is elsewhere), then
  // falls back to all open .pivotal documents with a QuickPick if needed.
  async function _resolvePivotalFile(): Promise<string | undefined> {
    // Prefer the last focused .pivotal editor
    const candidate = _lastPivotalEditor ?? vscode.window.activeTextEditor;
    if (candidate && (
      candidate.document.languageId === 'pivotal' ||
      candidate.document.uri.fsPath.endsWith('.pivotal')
    )) {
      await candidate.document.save();
      return candidate.document.uri.fsPath;
    }
    // Fall back: collect all open .pivotal documents
    const openPivotal = vscode.workspace.textDocuments.filter(
      d => d.uri.fsPath.endsWith('.pivotal') && !d.isUntitled
    );
    if (!openPivotal.length) {
      vscode.window.showErrorMessage('Pivotal: No .pivotal file is open.');
      return undefined;
    }
    if (openPivotal.length === 1) {
      await openPivotal[0].save();
      return openPivotal[0].uri.fsPath;
    }
    // Multiple open — ask the user
    const pick = await vscode.window.showQuickPick(
      openPivotal.map(d => ({ label: path.basename(d.uri.fsPath), description: d.uri.fsPath, doc: d })),
      { title: 'Pivotal: Select file to compile', placeHolder: 'Choose a .pivotal file' },
    );
    if (!pick) { return undefined; }
    await pick.doc.save();
    return pick.doc.uri.fsPath;
  }

  // --- Helper: get the Python executable for compile/export ---
  // Priority: 1) pivotal.pythonPath setting  2) active Jupyter kernel interpreter
  //           3) python.defaultInterpreterPath  4) "python"
  async function _getPythonPath(): Promise<string> {
    // 1. Explicit Pivotal setting
    const pivotalCfg = vscode.workspace.getConfiguration('pivotal');
    const explicit = pivotalCfg.get<string>('pythonPath');
    if (explicit && explicit.trim() && !explicit.includes('${')) {
      return explicit.trim();
    }

    // 2. Active Jupyter kernel interpreter (ms-toolsai.jupyter extension API)
    try {
      const jupyterExt = vscode.extensions.getExtension('ms-toolsai.jupyter');
      if (jupyterExt) {
        const api = await jupyterExt.activate() as Record<string, unknown>;
        // Jupyter extension exposes kernels.getKernel() for a notebook document
        const kernels = api.kernels as Record<string, unknown> | undefined;
        if (kernels) {
          // Try to find the active kernel's interpreter path from open notebooks
          const getKernel = kernels.getKernel as ((nb: unknown) => unknown) | undefined;
          if (getKernel) {
            for (const nb of vscode.workspace.notebookDocuments) {
              const kernel = getKernel.call(kernels, nb);
              if (kernel) {
                const k = kernel as Record<string, unknown>;
                const kSpec = k.kernelConnectionMetadata as Record<string, unknown> | undefined;
                const interpPath = kSpec?.interpreter as Record<string, unknown> | undefined;
                const execPath = interpPath?.path as string | undefined;
                if (execPath && execPath.trim() && !execPath.includes('${')) {
                  return execPath.trim();
                }
              }
            }
          }
        }
      }
    } catch { /* Jupyter API not available or changed shape — fall through */ }

    // 3. VS Code Python extension selected interpreter
    const cfg = vscode.workspace.getConfiguration('python');
    const interp = cfg.get<string>('defaultInterpreterPath') || cfg.get<string>('pythonPath');
    // Ignore placeholder values that haven't been expanded
    if (interp && interp.trim() && !interp.includes('${')) {
      return interp.trim();
    }
    return 'python';
  }

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
    ' ', '\t', ',',
  );

  // --- Command: Show Pivotal Viewer panel ---
  const showViewer = vscode.commands.registerCommand('pivotal.showViewer', () => {
    _getOrCreateViewerPanel(context, true);
  });

  // --- Phase 4: Explorer (single unified view) ---
  _explorerProvider = new _ExplorerProvider();
  _pivotalTreeView = vscode.window.createTreeView('pivotalExplorer', {
    treeDataProvider: _explorerProvider,
    // filterOnType was added in VS Code 1.73; cast to avoid old @types/vscode complaint
    ...(({ filterOnType: true, showCollapseAll: true }) as Record<string, unknown>),
  } as Parameters<typeof vscode.window.createTreeView>[1]) as vscode.TreeView<ExplorerNode>;
  context.subscriptions.push(_pivotalTreeView!);

  // --- Command: View item (focus in viewer panel) ---
  const explorerView = vscode.commands.registerCommand(
    'pivotal.explorer.view',
    (node: ExplorerNode) => {
      if (node.kind !== 'item') { return; }
      const panel = _getOrCreateViewerPanel(context, true);
      panel.webview.postMessage({ type: 'focus', name: node.name });
    },
  );

  // --- Command: Click a column in the explorer → scroll viewer to that column ---
  const explorerScrollToColumn = vscode.commands.registerCommand(
    'pivotal.explorer.scrollToColumn',
    (node: ExplorerNode) => {
      if (node.kind !== 'column') { return; }
      // First focus the parent dataframe in the viewer, then scroll to the column
      const panel = _getOrCreateViewerPanel(context, true);
      panel.webview.postMessage({ type: 'focus', name: node.parent });
      // Small delay so the grid is rendered before we scroll
      setTimeout(() => {
        panel.webview.postMessage({ type: 'scroll_col', name: node.parent, col: node.col });
      }, 50);
    },
  );

  // --- Command: Quick-open a viewer item by name (keyboard shortcut from editor) ---
  const quickOpen = vscode.commands.registerCommand(
    'pivotal.quickOpen',
    async () => {
      if (_explorerItems.size === 0) {
        vscode.window.showInformationMessage('No Pivotal data loaded yet.');
        return;
      }
      const iconMap: Record<string, string> = { dataframe: '$(table)', chart: '$(graph)', gt_table: '$(list-flat)' };
      const picks = [..._explorerItems.entries()].map(([name, p]) => {
        const t = p.type as string;
        const shape = t === 'dataframe' && Array.isArray((p as any).shape)
          ? `${((p as any).shape as [number,number])[0].toLocaleString()} × ${((p as any).shape as [number,number])[1]}`
          : t === 'chart' ? 'chart' : 'table';
        return { label: `${iconMap[t] ?? '$(database)'}  ${name}`, description: shape, name };
      });
      const chosen = await vscode.window.showQuickPick(picks, {
        placeHolder: 'Type to search data, charts and tables…',
        matchOnDescription: false,
      });
      if (!chosen) { return; }
      // Reveal in explorer tree (without stealing focus from editor)
      const node = { kind: 'item' as const, type: _explorerItems.get(chosen.name)?.type as string, name: chosen.name, payload: _explorerItems.get(chosen.name)! };
      _pivotalTreeView?.reveal(node, { select: true, focus: false, expand: true }).then(undefined, () => {});
      // Show in viewer
      const panel = _getOrCreateViewerPanel(context, true);
      panel.webview.postMessage({ type: 'focus', name: chosen.name });
    },
  );

  // --- Command: Search / filter explorer items ---
  const explorerSearch = vscode.commands.registerCommand(
    'pivotal.explorer.search',
    async () => {
      const current = _explorerProvider?.filter ?? '';
      const value = await vscode.window.showInputBox({
        prompt: 'Filter Pivotal Explorer items',
        placeHolder: 'Type to filter by name… (leave empty to clear)',
        value: current,
      });
      if (value === undefined || !_explorerProvider) { return; }
      _explorerProvider.filter = value.toLowerCase().trim();
      _refreshExplorer();
    },
  );

  // --- Command: Delete item (from explorer inline button) ---
  const explorerDelete = vscode.commands.registerCommand(
    'pivotal.explorer.delete',
    (node: ExplorerNode) => {
      if (node.kind !== 'item') { return; }
      const { name } = node;
      sendToBridge({ type: 'delete', name });
      _explorerItems.delete(name);
      _refreshExplorer();
      _viewerPanel?.webview.postMessage({ type: 'delete', name });
    },
  );

  // --- Phase 5: Load Dataset ---
  const loadDataset = vscode.commands.registerCommand('pivotal.loadDataset', async () => {
    const uris = await vscode.window.showOpenDialog({
      canSelectMany: false,
      filters: { 'Data files': ['csv', 'xlsx', 'parquet', 'json'], 'All files': ['*'] },
      title: 'Pivotal: Select data file',
    });
    if (!uris?.length) { return; }
    const filePath = uris[0].fsPath;
    const defaultName = path.basename(filePath).replace(/\.[^.]+$/, '').replace(/\W+/g, '_').replace(/^(\d)/, '_$1');
    const tableName = await vscode.window.showInputBox({
      prompt: 'Table name',
      value: defaultName,
      validateInput: v => /^\w+$/.test(v) ? null : 'Use letters, numbers and underscores only',
    });
    if (!tableName) { return; }
    const editor = vscode.window.activeTextEditor;
    if (!editor) { vscode.window.showErrorMessage('Pivotal: No active editor.'); return; }
    const code = `load ${tableName} "${filePath}"\n`;
    editor.edit(eb => eb.insert(editor.selection.active, code));
  });

  // --- Phase 5: Save Package ---
  const savePackage = vscode.commands.registerCommand('pivotal.savePackage', async () => {
    const allItems = [..._explorerItems.values()].map(p => ({
      label: p.name as string,
      description: p.type as string,
      picked: true,
    }));
    if (!allItems.length) {
      vscode.window.showInformationMessage('Pivotal: No data in session to save — run a Pivotal cell first.');
      return;
    }
    const picks = await vscode.window.showQuickPick(allItems, {
      canPickMany: true,
      title: 'Pivotal: Select items to save',
      placeHolder: 'Choose dataframes, charts, or tables to include',
    });
    if (!picks?.length) { return; }

    const dirUris = await vscode.window.showOpenDialog({
      canSelectFiles: false, canSelectFolders: true, canSelectMany: false,
      title: 'Pivotal: Select output directory',
    });
    if (!dirUris?.length) { return; }
    const outDir = dirUris[0].fsPath;

    const pkgName = await vscode.window.showInputBox({
      prompt: 'Package name (becomes the sub-folder)',
      value: 'output',
      validateInput: v => /^\w+$/.test(v) ? null : 'Use letters, numbers and underscores only',
    });
    if (!pkgName) { return; }

    const fmt = await vscode.window.showQuickPick(['parquet', 'csv', 'xlsx'], {
      title: 'Pivotal: Output format',
      placeHolder: 'Select file format',
    });
    if (!fmt) { return; }

    const editor = vscode.window.activeTextEditor;
    if (!editor) { vscode.window.showErrorMessage('Pivotal: No active editor.'); return; }

    const includeList = picks.map(p => p.label).join(', ');
    const sep = outDir.includes('\\') ? '\\\\' : '/';
    const pkgPath = outDir + sep + pkgName;
    const code = `save "${pkgPath}"\n    format ${fmt}\n    include ${includeList}\n`;
    editor.edit(eb => eb.insert(editor.selection.active, code));
  });

  // --- Phase 5: Compile to Python or SQL ---
  // Merged replacement for the old compileToFile + codeExport commands.
  // Uses _resolvePivotalFile() so it works regardless of where focus currently is.
  const codeExport = vscode.commands.registerCommand('pivotal.codeExport', async () => {
    const filePath = await _resolvePivotalFile();
    if (!filePath) { return; }

    const backend = await vscode.window.showQuickPick(
      ['pandas', 'polars', 'duckdb', 'sql'],
      { title: 'Pivotal: Compile — select backend', placeHolder: 'Target backend' },
    );
    if (!backend) { return; }

    const pythonPath = await _getPythonPath();
    const outPath = filePath.replace(/\.pivotal$/, backend === 'sql' ? '.sql' : '.py');

    return new Promise<void>(resolve => {
      exec(`"${pythonPath}" -m pivotal --compile --backend ${backend} "${filePath}"`,
        (error, _stdout, stderr) => {
          if (error) {
            const msg = stderr || error.message;
            const isNoModule = msg.includes('No module named');
            vscode.window.showErrorMessage(
              `Pivotal compile error: ${msg}` +
              (isNoModule ? `\n\nUsing: ${pythonPath}\nTip: Set "pivotal.pythonPath" in settings to point to the environment where Pivotal is installed.` : ''),
              ...(isNoModule ? ['Configure Python Path', 'Select VS Code Interpreter'] : [])
            ).then(action => {
              if (action === 'Configure Python Path') {
                vscode.commands.executeCommand('pivotal.selectPythonPath');
              } else if (action === 'Select VS Code Interpreter') {
                vscode.commands.executeCommand('python.setInterpreter');
              }
            });
          } else {
            vscode.window.showInformationMessage(`Pivotal: Compiled to ${outPath}`, 'Open File')
              .then(action => {
                if (action === 'Open File') {
                  vscode.workspace.openTextDocument(outPath).then(doc =>
                    vscode.window.showTextDocument(doc)
                  );
                }
              });
          }
          resolve();
        }
      );
    });
  });

  // --- Command: Select Python environment for Pivotal compile/export ---
  const selectPythonPath = vscode.commands.registerCommand('pivotal.selectPythonPath', async () => {
    const current = vscode.workspace.getConfiguration('pivotal').get<string>('pythonPath') || '';
    const detected = await _getPythonPath();
    const value = await vscode.window.showInputBox({
      title: 'Pivotal: Select Python Environment',
      prompt: 'Enter the full path to the Python executable for the environment where Pivotal is installed',
      value: current || detected,
      placeHolder: '/home/user/miniconda3/envs/myenv/bin/python',
      validateInput: v => {
        if (!v || !v.trim()) { return 'Path cannot be empty'; }
        return undefined;
      },
    });
    if (value === undefined) { return; } // cancelled
    await vscode.workspace.getConfiguration('pivotal').update(
      'pythonPath', value.trim(), vscode.ConfigurationTarget.Global,
    );
    vscode.window.showInformationMessage(`Pivotal: Python path set to ${value.trim()}`);
  });

  // Helper: execute a Pivotal code string directly in the interactive window.
  // jupyter.execSelectionInteractive uses activeTextEditor to resolve the kernel.
  // We briefly focus the .pivotal editor to set activeTextEditor, then the caller's
  // sourcePanel.reveal() immediately returns focus to the GUI.
  async function _executeGuiCode(code: string): Promise<void> {
    const escaped = JSON.stringify(code);
    const cellText = `import pivotal; get_ipython().run_cell_magic('pivotal', '', ${escaped})`;
    try {
      if (_lastPivotalEditor) {
        await vscode.window.showTextDocument(
          _lastPivotalEditor.document,
          { viewColumn: _lastPivotalEditor.viewColumn, preserveFocus: false, preview: false },
        );
      }
      await vscode.commands.executeCommand('jupyter.execSelectionInteractive', cellText);
    } catch {
      vscode.window.showErrorMessage(
        'Pivotal: Could not send to Interactive Window — ensure a Python kernel is running.'
      );
    }
  }

  // Track the last focused .pivotal text editor so Insert works even when the
  // GUI webview panel has keyboard focus (webviews do not preserve activeTextEditor).
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(editor => {
      if (editor && (
        editor.document.languageId === 'pivotal' ||
        editor.document.uri.fsPath.endsWith('.pivotal')
      )) {
        _lastPivotalEditor = editor;
      }
    }),
  );
  // Seed with whatever is already open.
  if (vscode.window.activeTextEditor?.document.languageId === 'pivotal') {
    _lastPivotalEditor = vscode.window.activeTextEditor;
  }

  // Helper: handle GUI actions (insert / execute).
  function _handleGuiAction(action: string, code: string, sourcePanel: vscode.WebviewPanel): void {
    if (action === 'insert') {
      const editor = _lastPivotalEditor ?? vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage(
          'Pivotal: No active .pivotal editor — open a .pivotal file and click in it first.'
        );
        return;
      }
      const wsEdit = new vscode.WorkspaceEdit();
      wsEdit.insert(editor.document.uri, editor.selection.active, code + '\n');
      vscode.workspace.applyEdit(wsEdit).then(() => {
        // Focus editor so user can see where the code was inserted.
        vscode.window.showTextDocument(editor.document, {
          viewColumn: editor.viewColumn, preserveFocus: false, preview: false,
        });
      });
    }
    if (action === 'execute') {
      _executeGuiCode(code).then(() => {
        // Refocus the GUI panel — jupyter.execSelectionInteractive steals focus.
        sourcePanel.reveal(undefined, false);
      });
    }
  }

  // --- Phase 5: Plot GUI ---
  async function _openGuiPanel(
    existing: vscode.WebviewPanel | null,
    viewType: string,
    title: string,
    buildHtml: () => string,
    onDispose: () => void,
  ): Promise<vscode.WebviewPanel> {
    if (existing) {
      existing.reveal(undefined, false);
      _sendTablesToGui(existing);
      return existing;
    }
    const guiColumn = _lastPivotalEditor?.viewColumn ?? vscode.ViewColumn.One;
    // Create with preserveFocus:false so the panel is active — needed for moveEditorToBelowGroup
    const panel = vscode.window.createWebviewPanel(
      viewType, title,
      { viewColumn: guiColumn, preserveFocus: false },
      { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] },
    );
    panel.webview.html = buildHtml();
    _sendTablesToGui(panel);
    panel.webview.onDidReceiveMessage((msg: Record<string, unknown>) => {
      _handleGuiAction(msg.action as string, msg.code as string, panel);
    }, undefined, context.subscriptions);
    panel.onDidDispose(onDispose, undefined, context.subscriptions);

    // Move the panel to a horizontal split below the .pivotal editor
    await vscode.commands.executeCommand('workbench.action.moveEditorToBelowGroup');

    // Restore focus to the .pivotal editor
    if (_lastPivotalEditor) {
      await vscode.window.showTextDocument(_lastPivotalEditor.document, {
        viewColumn: guiColumn, preserveFocus: false, preview: false,
      });
    }

    return panel;
  }

  const plotGui = vscode.commands.registerCommand('pivotal.plotGui', async () => {
    _plotGuiPanel = await _openGuiPanel(
      _plotGuiPanel, 'pivotalPlotGui', 'Pivotal: Plot',
      _buildPlotGuiHtml,
      () => { _plotGuiPanel = null; },
    );
  });

  // --- Phase 5: Pivot GUI ---
  const pivotGui = vscode.commands.registerCommand('pivotal.pivotGui', async () => {
    _pivotGuiPanel = await _openGuiPanel(
      _pivotGuiPanel, 'pivotalPivotGui', 'Pivotal: Pivot',
      _buildPivotGuiHtml,
      () => { _pivotGuiPanel = null; },
    );
  });

  // --- Bridge handler: forward Python viewer messages to viewer + explorer ---
  onBridgeMessage((msg: Record<string, unknown>) => {
    const t = msg.type as string;
    if (t === 'dataframe' || t === 'chart' || t === 'gt_table') {
      const isFirstItem = _explorerItems.size === 0;
      _explorerItems.set(msg.name as string, msg);
      _refreshExplorer(t);
      _refreshGuiPanels();
      if (isFirstItem) {
        vscode.commands.executeCommand('pivotalExplorer.focus');
      }
      // Ensure the viewer panel exists.  If the webview has already
      // confirmed 'ready', post immediately; otherwise the item is stored
      // in _explorerItems and will be replayed when 'ready' fires.
      const panel = _getOrCreateViewerPanel(context);
      if (_viewerReady) {
        panel.webview.postMessage(msg);
      }
    } else if (t === 'delete') {
      const existing = _explorerItems.get(msg.name as string);
      _explorerItems.delete(msg.name as string);
      _refreshExplorer(existing?.type as string | undefined);
      _refreshGuiPanels();
      _viewerPanel?.webview.postMessage(msg);
    } else if (t === 'clear') {
      _explorerItems.clear();
      _refreshExplorer();
      _refreshGuiPanels();
      _viewerPanel?.webview.postMessage(msg);
    } else if (t === 'focus') {
      _viewerPanel?.webview.postMessage(msg);
    }
  });

  _startBridgeWatcher(context);

  // ── Interpreter sync ─────────────────────────────────────────────────────
  // When pivotal.pythonPath is configured, write it into python.defaultInterpreterPath
  // (workspace-level) so VS Code Jupyter picks the right kernel for new sessions.
  // This runs on activation and whenever the setting changes.

  function _syncPivotalInterpreter(): void {
    const pivotalPath = vscode.workspace.getConfiguration('pivotal').get<string>('pythonPath');
    if (!pivotalPath || !pivotalPath.trim() || pivotalPath.includes('${')) { return; }
    const resolved = pivotalPath.trim();
    const pythonCfg = vscode.workspace.getConfiguration('python');
    const current = pythonCfg.get<string>('defaultInterpreterPath') || '';
    if (current === resolved) { return; }   // already in sync
    const target = vscode.workspace.workspaceFolders?.length
      ? vscode.ConfigurationTarget.Workspace
      : vscode.ConfigurationTarget.Global;
    pythonCfg.update('defaultInterpreterPath', resolved, target);
  }

  _syncPivotalInterpreter();

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(e => {
      if (e.affectsConfiguration('pivotal.pythonPath')) {
        _syncPivotalInterpreter();
      }
    }),
  );

  // ── Wrong-kernel warning ─────────────────────────────────────────────────
  // After each execution attempt, if the bridge hasn't connected within a few
  // seconds and pivotal.pythonPath is configured, the kernel is probably wrong.
  // Show a one-shot notification so the user knows what to fix.

  let _kernelWarnShown = false;

  function _scheduleKernelCheck(): void {
    const pivotalPath = vscode.workspace.getConfiguration('pivotal').get<string>('pythonPath');
    if (!pivotalPath || !pivotalPath.trim() || _kernelWarnShown || _bridgeSocket) { return; }
    setTimeout(() => {
      // Re-check: did the bridge connect in the meantime?
      if (_bridgeSocket || _kernelWarnShown) { return; }
      _kernelWarnShown = true;
      vscode.window.showWarningMessage(
        `Pivotal: the interactive kernel may not be using the configured Python environment (${pivotalPath.trim()}). If results are not appearing, switch the kernel and restart.`,
        'Switch Kernel',
        'Dismiss',
      ).then(choice => {
        if (choice === 'Switch Kernel') {
          vscode.commands.executeCommand('jupyter.selectKernelForInteractiveWindow');
        }
      });
    }, 8000);
  }

  // Reset the warning flag when the bridge connects (kernel is correct).
  onBridgeMessage(() => { _kernelWarnShown = false; });

  context.subscriptions.push(
    executeFile,
    executeInNotebook,
    executeSelectionInNotebook,
    hoverProvider,
    completionProvider,
    showViewer,
    explorerView,
    explorerDelete,
    explorerSearch,
    explorerScrollToColumn,
    quickOpen,
    loadDataset,
    savePackage,
    codeExport,
    selectPythonPath,
    plotGui,
    pivotGui,
  );
}

export function deactivate(): void { /* nothing to clean up */ }
