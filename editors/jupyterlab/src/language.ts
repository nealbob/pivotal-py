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
  meta:     tags.meta,   // :variable substitutions
};

interface PivotalState {
  inBlockComment: boolean;
  inString: string | false; // the quote character, or false
}

const keywords: Record<string, true> = {
  load: true, df: true, filter: true,
  select: true, sort: true, order: true, by: true,
  group: true, merge: true, pivot: true, unpivot: true, apply: true,
  plot: true, python: true, from: true, where: true,
  as: true, on: true, rows: true, cols: true, assign: true,
  agg: true, drop: true, fillna: true, dropna: true, delete: true,
  distinct: true, concat: true, rename: true, save: true,
  table: true, stub: true, col: true, stripe: true,
  rank: true, lag: true, lead: true,
  cumsum: true, cummean: true, cummin: true, cummax: true, rolling: true,
  id: true, variable: true, value: true,
};

const builtins: Record<string, true> = {
  mean: true, min: true, max: true, sum: true,
  count: true, avg: true, median: true, std: true,
  asc: true, desc: true, pct: true, left: true, right: true,
  inner: true, outer: true,
  // word-operator comparators
  in: true, not: true, between: true,
  contains: true, startswith: true, endswith: true,
  and: true, or: true,
  integer: true, currency: true, percent: true, date: true, number: true,
};

const atoms: Record<string, true> = {
  True: true, False: true, true: true, false: true,
  None: true, none: true,
};

function tokenize(stream: StringStream, state: PivotalState): string | null {
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
    return { inBlockComment: false, inString: false };
  },

  token(stream, state) {
    return tokenize(stream, state);
  },

  // Map our string token names to CM6 highlight tags
  // (StreamLanguage uses the tag table from tokenTable via languageHighlighting)
  tokenTable,
});
