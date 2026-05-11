import { StreamLanguage, StringStream } from '@codemirror/language';
import { tags } from '@lezer/highlight';

// Token type → CM6 highlight tag
const tokenTable: Record<string, any> = {
  keyword:  tags.keyword,
  builtin:  tags.standard(tags.name),
  atom:     tags.atom,
  string:   tags.string,
  comment:  tags.comment,
  number:   tags.number,
  operator: tags.operator,
  variable: tags.variableName,
  loopVariable: tags.variableName,
  meta:     tags.meta,   // :variable substitutions
};

interface PivotalState {
  inBlockComment: boolean;
  inString: string | false; // the quote character, or false
  loopVars: { indent: number; name: string }[];
}

const keywords: Record<string, true> = {
  load: true,
  bulk: true,
  with: true,
  from: true,
  for: true,
  function: true,
  return: true,
  list: true,
  filter: true,
  assert: true,
  check: true,
  select: true,
  sort: true,
  order: true,
  group: true,
  merge: true,
  pivot: true,
  python: true,
  plot: true,
  drop: true,
  fillna: true,
  dropna: true,
  distinct: true,
  concat: true,
  intersect: true,
  exclude: true,
  rename: true,
  round: true,
  apply: true,
  save: true,
  end: true,
  delete: true,
  show: true,
  unpivot: true,
  rolling: true,
  rank: true,
  lag: true,
  lead: true,
  cumsum: true,
  cummean: true,
  cummin: true,
  cummax: true,
  cast: true,
  agg: true,
  table: true,
  all: true,
  query: true,
  where: true,
  else: true,
  as: true,
  on: true,
  by: true,
  rows: true,
  cols: true,
  head: true,
  summary: true,
  strict: true,
  stub: true,
  format: true,
  id: true,
  variable: true,
  value: true,
  left_on: true,
  right_on: true,
  suffixes: true,
  header: true,
  names: true,
  path: true,
  include: true,
  chart_format: true,
  title: true,
  subtitle: true,
  font: true,
  size: true,
  spanner: true,
  auto: true,
  label: true,
  stripe: true,
  canvas: true,
  style: true,
  kind: true,
  colormap: true,
  legend: true,
  x: true,
  y: true,
  c: true,
  source: true,
  column: true,
  min_periods: true,
};

const builtins: Record<string, true> = {
  mean: true,
  min: true,
  max: true,
  sum: true,
  count: true,
  avg: true,
  median: true,
  std: true,
  nunique: true,
  wavg: true,
  wmean: true,
  upper: true,
  lower: true,
  trim: true,
  ltrim: true,
  rtrim: true,
  left: true,
  right: true,
  substr: true,
  len: true,
  replace: true,
  regex_extract: true,
  regex_replace: true,
  year: true,
  month: true,
  day: true,
  quarter: true,
  dayofweek: true,
  hour: true,
  minute: true,
  date_format: true,
  to_date: true,
  date_diff: true,
  date_add: true,
  int: true,
  integer: true,
  float: true,
  string: true,
  str: true,
  bool: true,
  boolean: true,
  datetime: true,
  inner: true,
  outer: true,
  asc: true,
  desc: true,
  pct: true,
  and: true,
  or: true,
  not: true,
  in: true,
  unique: true,
  null: true,
  between: true,
  contains: true,
  matches: true,
  startswith: true,
  endswith: true,
  number: true,
  currency: true,
  percent: true,
  date: true,
};

const atoms: Record<string, true> = {
  True: true,
  False: true,
  true: true,
  false: true,
  None: true,
  none: true,
};

function tokenize(stream: StringStream, state: PivotalState): string | null {
  if (stream.sol()) {
    if (!/^\s*$/.test(stream.string)) {
      const indent = stream.string.match(/^\s*/)?.[0].length ?? 0;
      while (state.loopVars.length && indent <= state.loopVars[state.loopVars.length - 1].indent) {
        state.loopVars.pop();
      }
    }
  }

  // --- Block comment continuation ---
  if (state.inBlockComment) {
    if (stream.match(/.*?\*\//)) {
      state.inBlockComment = false;
    } else {
      stream.skipToEnd();
    }
    return 'comment';
  }

  // --- Block comment start ---
  if (stream.match('/*')) {
    if (stream.match(/.*?\*\//)) {
      // closed on the same line
    } else {
      state.inBlockComment = true;
      stream.skipToEnd();
    }
    return 'comment';
  }

  // --- Line comments: # and -- ---
  if (stream.match(/#.*/) || stream.match(/--.*$/)) {
    return 'comment';
  }

  // --- Strings ---
  if (stream.match(/"(?:[^\\"]|\\.)*"/)) return 'string';
  if (stream.match(/'(?:[^\\']|\\.)*'/)) return 'string';

  // --- Python variable substitution :identifier ---
  if (stream.match(/:[a-zA-Z][a-zA-Z0-9_]*/)) return 'meta';

  // --- Numbers ---
  if (stream.match(/\d+(\.\d+)?/)) return 'number';

  // --- Identifiers, keywords, builtins, atoms ---
  if (stream.match(/[a-zA-Z_][a-zA-Z0-9_]*/)) {
    const word = stream.current();
    const before = stream.string.slice(0, stream.start);
    const forHeader = stream.string.match(/^(\s*)for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\b/);
    if (/^\s*for\s+$/.test(before)) {
      const indent = before.length - before.trimStart().length;
      if (!state.loopVars.some(v => v.indent === indent && v.name === word)) {
        state.loopVars.push({ indent, name: word });
      }
      return 'loopVariable';
    }
    if (forHeader && forHeader[2] === word && state.loopVars.some(v => v.name === word)) {
      return 'loopVariable';
    }
    if (state.loopVars.some(v => v.name === word)) return 'loopVariable';
    if (Object.prototype.hasOwnProperty.call(keywords, word))  return 'keyword';
    if (Object.prototype.hasOwnProperty.call(builtins, word))  return 'builtin';
    if (Object.prototype.hasOwnProperty.call(atoms, word))     return 'atom';
    return 'variable';
  }

  // --- Operators ---
  if (stream.match(/==|!=|>=|<=|>|<|\*\*|[+\-*/=]/)) return 'operator';

  // Consume unrecognised characters one at a time
  stream.next();
  return null;
}

export const pivotalLanguage = StreamLanguage.define<PivotalState>({
  name: 'pivotal',

  startState(): PivotalState {
    return { inBlockComment: false, inString: false, loopVars: [] };
  },

  token(stream, state) {
    return tokenize(stream, state);
  },

  // Map our string token names to CM6 highlight tags
  // (StreamLanguage uses the tag table from tokenTable via languageHighlighting)
  tokenTable,
});
