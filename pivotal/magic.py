try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
    from IPython.display import display
    import pandas as pd
except ImportError:
    # Fallback if IPython is not installed
    class Magics: pass
    def magics_class(cls): return cls
    def cell_magic(func): return func
    def line_magic(func): return func
    display = print

from .dsl_parser import DSLParser

@magics_class
class PivotalMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.parser = DSLParser()
        self.auto_pivotal = False
        
        # Register input transformer
        if hasattr(shell, 'input_transformers_cleanup'):
            shell.input_transformers_cleanup.append(self.input_transform)
            
        # Register completer (not supported in all environments, e.g. VS Code notebooks)
        try:
            self.shell.set_hook('complete_command', self.pivotal_completer, re_key='%%pivotal')
            self.shell.set_hook('complete_command', self.pivotal_completer, str_key='%pivotal_auto')
        except Exception:
            pass
        
        # Try to register a custom completer for auto mode
        try:
            from IPython.core.completer import Completer
            # We can't easily inject into the main completer logic without being a proper extension
            # But we can try to add a hook that runs for all cells if auto_pivotal is on
            pass
        except ImportError:
            pass

    def pivotal_completer(self, event):
        """Custom completer for Pivotal DSL"""
        # Basic keywords list
        keywords = [
            'load', 'df', 'filter', 'select', 'sort',
            'group by', 'merge', 'pivot', 'plot', 'python', 'apply',
            'drop', 'dropna', 'fillna', 'distinct', 'concat', 'rename',
            'mean', 'min', 'max', 'sum', 'count', 'avg', 'median', 'std',
            'asc', 'desc', 'left', 'right', 'inner', 'outer',
            'from', 'as', 'on', 'rows', 'cols',
            'between', 'contains', 'not contains', 'startswith', 'endswith',
        ]
        
        # Add table names and columns from parser state
        if hasattr(self.parser, 'table_info'):
            for table_name, info in self.parser.table_info.items():
                keywords.append(table_name)
                if 'columns' in info:
                    # Flatten columns if they are lists (MultiIndex)
                    for col in info['columns']:
                        if isinstance(col, list):
                            keywords.extend([str(c) for c in col])
                        else:
                            keywords.append(str(col))
        
        # Filter matches
        return [k for k in keywords if k.startswith(event.symbol)]

    @line_magic
    def pivotal_auto(self, line):
        """Toggle automatic Pivotal parsing for all cells."""
        self.auto_pivotal = not self.auto_pivotal
        print(f"Automatic Pivotal parsing is now {'ON' if self.auto_pivotal else 'OFF'}")

    def input_transform(self, lines):
        """
        Transform input if auto_pivotal is enabled and code parses as Pivotal.
        """
        if not self.auto_pivotal:
            return lines
        
        # Join lines to check content
        code = ''.join(lines)
        
        # Ignore magics
        if code.strip().startswith('%'):
            return lines
            
        # Try to parse
        results = self.parser.parse(code)
        
        if isinstance(results, dict) and 'error' in results:
            # Parse failed, assume it's Python
            return lines
            
        # If successful, generate code
        try:
            python_code_list = self.parser.generate_code(results)
            
            # Process code to add display logic
            final_lines = []
            for block in python_code_list:
                # Split block into lines
                block_lines = block.splitlines()
                for line in block_lines:
                    if '__table_name__ =' in line:
                        # Extract table name
                        # Line format: __table_name__ = 'df'
                        try:
                            parts = line.split('=')
                            t_name = parts[1].strip().strip("'").strip('"')
                            # Add display logic
                            # We use display() if available, else print
                            display_code = f"try:\n    from IPython.display import display\n    print(f\"Table '{t_name}' shape: {{{t_name}.shape}}\")\n    display({t_name}.head())\nexcept:\n    print({t_name}.head())"
                            final_lines.extend(display_code.split('\n'))
                        except:
                            pass
                    elif '#__pivotal__' in line:
                        continue
                    else:
                        final_lines.append(line)
                final_lines.append("") # Add newline between blocks
                
            return [l + '\n' for l in final_lines]
        except Exception:
            return lines

    @cell_magic
    def pivotal(self, line, cell):
        """
        Execute Pivotal DSL code.
        Usage:
            %%pivotal
            load df "data.csv"
            filter df
                col > 5
        """
        # Ensure pandas is imported in the user namespace
        self.shell.push({'pd': pd})

        if not cell.endswith('\n'):
            cell += '\n'

        results = self.parser.parse(cell)

        if isinstance(results, dict) and 'error' in results:
            print(f"Pivotal Parse Error: {results['error']}")
            return

        python_code_list = self.parser.generate_code(results)
        
        for i, code_block in enumerate(python_code_list):
            # Execute the code in the shell's user namespace
            result = self.shell.run_cell(code_block)
            
            if result.error_in_exec:
                return # Stop on error

            # Try to display the table info if a table was created/modified
            # We look at the AST result to see which table was affected
            if i < len(results) and 'table_name' in results[i]:
                table_name = results[i]['table_name']
                # Check if it exists in user_ns
                if table_name in self.shell.user_ns:
                    obj = self.shell.user_ns[table_name]
                    if isinstance(obj, pd.DataFrame):
                        print(f"Table '{table_name}' shape: {obj.shape}")
                        display(obj.head())

def load_ipython_extension(ipython):
    ipython.register_magics(PivotalMagics)
