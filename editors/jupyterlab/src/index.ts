import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
} from '@jupyterlab/application';

import { IEditorLanguageRegistry } from '@jupyterlab/codemirror';

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@pivotal/jupyterlab:language',
  description: 'Syntax highlighting for the Pivotal data transformation DSL',
  autoStart: true,
  requires: [IEditorLanguageRegistry],
  activate: (_app: JupyterFrontEnd, languages: IEditorLanguageRegistry) => {
    languages.addLanguage({
      name: 'pivotal',
      mime: 'text/x-pivotal',
      extensions: ['.pivotal'],
      // Lazy-load the StreamLanguage so it doesn't block JupyterLab startup
      load: async () => {
        const { pivotalLanguage } = await import('./language');
        return { support: pivotalLanguage };
      },
    });
  },
};

export default plugin;
