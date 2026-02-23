import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
} from '@jupyterlab/application';

import {
  IEditorLanguageRegistry,
  IEditorExtensionRegistry,
} from '@jupyterlab/codemirror';
import { LanguageSupport } from '@codemirror/language';
import { Compartment, Prec } from '@codemirror/state';
import { EditorView } from '@codemirror/view';

import { pivotalLanguage } from './language';

const MAGIC_RE = /^%%pivotal(\s|$)/;

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@pivotal/jupyterlab:language',
  description: 'Syntax highlighting for the Pivotal data transformation DSL',
  autoStart: true,
  requires: [IEditorLanguageRegistry, IEditorExtensionRegistry],
  activate: (
    _app: JupyterFrontEnd,
    languages: IEditorLanguageRegistry,
    extensions: IEditorExtensionRegistry
  ) => {
    // Register the language for standalone .pivotal files
    languages.addLanguage({
      name: 'pivotal',
      mime: 'text/x-pivotal',
      extensions: ['.pivotal'],
      load: async () => new LanguageSupport(pivotalLanguage),
    });

    // For notebook cells: watch each editor's first line and switch to pivotal
    // highlighting when it contains the %%pivotal cell magic.
    extensions.addExtension({
      name: '@pivotal/jupyterlab:magic-highlight',
      factory: () => {
        // Each editor instance gets its own Compartment via this closure.
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
                isPivotal ? Prec.highest(new LanguageSupport(pivotalLanguage)) : []
              ),
            });
          }),
        ];

        // instance(value) → initial CM6 extension for this editor.
        // reconfigure() → null because this extension has no user settings.
        return { instance: () => ext, reconfigure: () => null };
      },
    });
  },
};

export default plugin;
