import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

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

  // --- Command: Execute in Interactive Notebook ---
  // Reads the .pivotal file, opens a VS Code Python Interactive Window,
  // and sends the contents as a %%pivotal cell so DataFrames render interactively.
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
    const cellText = `%%pivotal\n${fileContents}`;

    // Open an Interactive Window (requires the Python + Jupyter VS Code extensions).
    // jupyter.createnewinteractive opens/focuses the interactive window.
    try {
      await vscode.commands.executeCommand('jupyter.createnewinteractive');
    } catch {
      // If the command doesn't exist, the Jupyter extension may not be installed.
      vscode.window.showErrorMessage(
        'Pivotal: Could not open Interactive Window. ' +
        'Make sure the Python and Jupyter VS Code extensions are installed.'
      );
      return;
    }

    // Small delay to let the Interactive Window initialise before sending code.
    await new Promise(resolve => setTimeout(resolve, 500));

    // jupyter.execSelectionInteractive runs the provided text in the active kernel.
    try {
      await vscode.commands.executeCommand('jupyter.execSelectionInteractive', cellText);
    } catch {
      vscode.window.showErrorMessage(
        'Pivotal: Failed to send code to Interactive Window. ' +
        'Please ensure a Python kernel is running.'
      );
    }
  });

  context.subscriptions.push(executeFile, executeInNotebook);
}

export function deactivate(): void { /* nothing to clean up */ }
