define(['base/js/namespace', 'codemirror/lib/codemirror'], function(Jupyter, CodeMirror) {
    "use strict";

    var load_ipython_extension = function() {
        console.log("Pivotal extension loading...");

        // 1. Define the CodeMirror mode for Pivotal DSL
        CodeMirror.defineMode("pivotal", function(config, parserConfig) {
            var keywords = {
                "load": true, "df": true, "dataframe": true, "filter": true,
                "select": true, "sort": true, "order": true, "by": true,
                "group": true, "merge": true, "join": true, "pivot": true,
                "plot": true, "python": true, "from": true, "where": true,
                "as": true, "on": true, "rows": true, "cols": true,
                "set": true
            };
            
            var builtins = {
                "mean": true, "min": true, "max": true, "sum": true,
                "count": true, "avg": true, "median": true, "std": true,
                "asc": true, "desc": true, "left": true, "right": true,
                "inner": true, "outer": true
            };

            var atoms = {
                "True": true, "False": true, "None": true
            };

            return {
                startState: function() {
                    return {inString: false};
                },
                token: function(stream, state) {
                    // Strings
                    if (stream.match(/"(?:[^\\"]|\\.)*"/)) return "string";
                    if (stream.match(/'(?:[^\\']|\\.)*'/)) return "string";

                    // Comments
                    if (stream.match(/#.*/)) return "comment";

                    // Numbers
                    if (stream.match(/\d+(\.\d+)?/)) return "number";

                    // Keywords and Identifiers
                    if (stream.match(/[a-zA-Z_][a-zA-Z0-9_]*/)) {
                        var word = stream.current();
                        if (keywords.hasOwnProperty(word)) return "keyword";
                        if (builtins.hasOwnProperty(word)) return "builtin";
                        if (atoms.hasOwnProperty(word)) return "atom";
                        return "variable";
                    }

                    // Operators and Punctuation
                    if (stream.match(/[+\-*\/=<>!]+/)) return "operator";
                    
                    stream.next();
                    return null;
                }
            };
        });

        CodeMirror.defineMIME("text/x-pivotal", "pivotal");

        // 2. Register the magic command highlighting
        // This tells Jupyter: if a cell starts with %%pivotal, use 'text/x-pivotal' mode
        if (Jupyter.CodeCell) {
            var modes = Jupyter.CodeCell.options_default.highlight_modes;
            modes['text/x-pivotal'] = {'reg': [/^%%pivotal/]};
        }

        // Apply to existing cells immediately
        if (Jupyter.notebook) {
            Jupyter.notebook.get_cells().forEach(function(cell) {
                if (cell.cell_type === 'code') {
                    cell.auto_highlight();
                }
            });
        }

        console.log("Pivotal extension loaded.");
    };

    return {
        load_ipython_extension: load_ipython_extension
    };
});
