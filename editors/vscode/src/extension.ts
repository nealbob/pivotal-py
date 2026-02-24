import * as vscode from 'vscode';
import * as fs from 'fs';

export function activate(context: vscode.ExtensionContext): void {

  // --- Command: Execute File via CLI ---
  // Runs `python -m pivotal <file>` in the integrated terminal.
  const executeFile = vscode.commands.registerCommand('pivotal.executeFile', () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('Pivotal: No active editor.');
      return;
    }

    const filePath = editor.document.uri.fsPath;
    if (!filePath.endsWith('.pivotal')) {
      vscode.window.showErrorMessage('Pivotal: Active file is not a .pivotal file.');
      return;
    }

    // Save the file before running
    editor.document.save().then(() => {
      const terminal = vscode.window.createTerminal('Pivotal');
      terminal.show(true); // preserve focus on editor
      terminal.sendText(`python -m pivotal "${filePath}"`);
    });
  });

  // --- Command: Execute File in Interactive Notebook ---
  // Reads the .pivotal file, opens a VS Code Python Interactive Window,
  // and sends the contents as %%pivotal cell(s) so DataFrames render interactively.
  // Supports #%% cell markers to split the file into separate cells.
  const executeInNotebook = vscode.commands.registerCommand('pivotal.executeInNotebook', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('Pivotal: No active editor.');
      return;
    }

    const filePath = editor.document.uri.fsPath;
    if (!filePath.endsWith('.pivotal')) {
      vscode.window.showErrorMessage('Pivotal: Active file is not a .pivotal file.');
      return;
    }

    // Save first
    await editor.document.save();

    const fileContents = fs.readFileSync(filePath, 'utf8');

    // Split on #%% markers so each section runs as its own cell.
    // If there are no markers the whole file is treated as one cell.
    const sections = fileContents.split(/^#%%[^\n]*$/m).map(s => s.trim()).filter(Boolean);

    // Open an Interactive Window (requires the Python + Jupyter VS Code extensions).
    try {
      await vscode.commands.executeCommand('jupyter.createnewinteractive');
    } catch {
      vscode.window.showErrorMessage(
        'Pivotal: Could not open Interactive Window. ' +
        'Make sure the Python and Jupyter VS Code extensions are installed.'
      );
      return;
    }

    // Small delay to let the Interactive Window initialise before sending code.
    await new Promise(resolve => setTimeout(resolve, 500));

    for (const section of sections) {
      // Use get_ipython().run_cell_magic() to avoid %%pivotal magic syntax issues.
      // JSON.stringify safely escapes newlines, quotes, and backslashes.
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

      // Brief pause between cells so the kernel isn't flooded.
      if (sections.length > 1) {
        await new Promise(resolve => setTimeout(resolve, 300));
      }
    }
  });

  // --- Command: Execute Selection in Interactive Notebook ---
  // Sends the currently selected text to the Interactive Window as a pivotal cell.
  // Works for any file type (.pivotal or .py with embedded %%pivotal cells).
  const executeSelectionInNotebook = vscode.commands.registerCommand(
    'pivotal.executeSelectionInNotebook',
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('Pivotal: No active editor.');
        return;
      }

      const selection = editor.selection;
      const selectedText = editor.document.getText(selection);

      if (!selectedText.trim()) {
        vscode.window.showInformationMessage('Pivotal: No text selected.');
        return;
      }

      // Open an Interactive Window if one is not already open.
      try {
        await vscode.commands.executeCommand('jupyter.createnewinteractive');
      } catch {
        vscode.window.showErrorMessage(
          'Pivotal: Could not open Interactive Window. ' +
          'Make sure the Python and Jupyter VS Code extensions are installed.'
        );
        return;
      }

      // Small delay to let the Interactive Window initialise.
      await new Promise(resolve => setTimeout(resolve, 500));

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

  context.subscriptions.push(executeFile, executeInNotebook, executeSelectionInNotebook);
}

export function deactivate(): void { /* nothing to clean up */ }
