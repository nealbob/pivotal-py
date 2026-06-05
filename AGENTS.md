# Agent Instructions

This file contains development guidance for AI agents and contributors working on Pivotal.

For user-facing Pivotal language syntax and examples, see `PIVOTAL.md`. Keep this file focused on development process, project structure, testing expectations, and release hygiene.

## Project Context

Pivotal is a pipeline-oriented data transformation DSL for Python and Jupyter. The package published to PyPI is `pivotal-lang`.

The implementation currently lives mostly in `pivotal/dsl_parser.py`, with supporting validation, magic commands, CLI helpers, and tests elsewhere in the repo.

## Important Files

- `PIVOTAL.md`: canonical user-facing syntax reference.
- `README.md`: PyPI/GitHub landing page and high-level examples.
- `CHANGELOG.md`: notable user-visible changes, especially anything needed for the next release notes.
- `pyproject.toml`: package metadata, dependencies, extras, and version.
- `pivotal/dsl_parser.py`: Lark grammar, transformer/AST handling, backend code generation, SQL CTE generation, parser execution/export logic.
- `pivotal/validator.py`: semantic validation and normalization after parsing, before code generation.
- `pivotal/magic.py`: Jupyter/IPython magic integration and widget-facing behavior.
- `pivotal/__main__.py`: command-line compile/export behavior.
- `pivotal/errors.py`: user-facing error translation and formatting.
- `editors/vscode/package.json`: VS Code extension metadata, commands, version, and build scripts.
- `editors/vscode/syntaxes/`: VS Code TextMate grammars for `.pivotal` files and `%%pivotal` Python injections.
- `editors/jupyterlab/package.json`: JupyterLab extension JavaScript package metadata and build scripts.
- `editors/jupyterlab/pyproject.toml`: Python package metadata for the `pivotal-lab` JupyterLab extension.
- `editors/jupyterlab/build.ps1`: Windows-friendly JupyterLab extension build script.
- `tests/test_commands.py`: core parser and pandas behavior tests.
- `tests/test_commands_polars.py`: Polars backend tests.
- `tests/test_commands_duckdb.py`: DuckDB backend tests.
- `tests/test_phase5_sql_cte.py`: SQL CTE backend tests.
- `tests/jupyter_demo_test.py`: Playwright-based Jupyter demo regression test.
- `docs/syntax/`: detailed syntax documentation by topic.
- `docs/jupyter.md`: Jupyter-specific documentation.

## When Changing The Grammar

If Pivotal syntax changes, check whether each surface below needs updating.

- Lark grammar: `grammar_indented` in `pivotal/dsl_parser.py`.
- Transformer / AST handling: `DSLTransformer` in `pivotal/dsl_parser.py`.
- Semantic validation and normalization: `pivotal/validator.py`.
- Backend code generation: `CodeGenerator` methods in `pivotal/dsl_parser.py`.
- Pandas backend behavior: `generate_*_pandas` methods in `pivotal/dsl_parser.py`.
- Polars backend behavior: `generate_*_polars` methods in `pivotal/dsl_parser.py`.
- DuckDB backend behavior: `generate_*_duckdb` methods in `pivotal/dsl_parser.py`.
- SQL CTE backend behavior: `generate_*_sql` methods and `_generate_code_sql` in `pivotal/dsl_parser.py`.
- Jupyter magic behavior, if syntax is exposed through notebooks: `pivotal/magic.py`.
- CLI compile/export behavior, if syntax affects file/notebook export: `pivotal/__main__.py`.
- Parser and pandas tests: `tests/test_commands.py`.
- Backend tests: `tests/test_commands_polars.py`, `tests/test_commands_duckdb.py`, and `tests/test_phase5_sql_cte.py`.
- Error tests: `tests/test_errors.py`.
- Syntax reference: `PIVOTAL.md`.
- User docs: relevant files under `docs/syntax/`, plus `docs/jupyter.md` when notebook behavior changes.
- README examples: `README.md`.
- Release notes: `CHANGELOG.md` under `Unreleased`.
- Shared editor/Pygments syntax tokens: `pivotal/syntax_tokens.json`; regenerate VS Code and JupyterLab syntax assets with `python scripts/generate_syntax_assets.py`.
- Editor extension builds: rebuild VS Code and JupyterLab extension artifacts after updating syntax highlighting, autocomplete, or grammar-adjacent editor behavior, then test the rebuilt extensions rather than only the source files.
- Binder/Jupyter demo repo: sibling checkout such as `C:\Code_win\pivotal-demo`.

## Testing Expectations

Run the smallest useful test set while developing, then broaden testing when the change touches shared syntax or backend behavior.

Useful commands:

```powershell
python -m pytest tests/test_commands.py
python -m pytest tests/test_commands_polars.py
python -m pytest tests/test_commands_duckdb.py
python -m pytest tests/test_phase5_sql_cte.py
python -m pytest
```

For grammar changes, prefer running the full test suite before finishing. If optional backend dependencies are missing, report which tests could not be run and why.

## Binder Demo Repo

The Binder demo for Pivotal in JupyterLab lives in a separate repository. The local development convention is to keep it beside this repo, for example:

- Main package repo: `C:\Code_win\pivotal-py`
- Demo repo: `C:\Code_win\pivotal-demo`

Grammar or user-facing behavior changes may require updating the demo repo so the Binder notebook still runs. When changing syntax, examples, Jupyter behavior, package startup, or dependencies, check whether the demo notebook and Binder config need corresponding changes.

## Running The Jupyter Demo Test

Use this when asked to run the demo notebook, test the Jupyter demo, check JupyterLab, verify the Pivotal demo, or run the notebook end-to-end.

1. Start JupyterLab from this repo:

```powershell
powershell -File C:\Code_win\pivotal-py\start_pivotal.ps1
```

2. Wait for a URL containing `http://127.0.0.1:8888/lab?token=...` and capture the full URL including the token.

3. If the script times out or no URL is printed, check for an already-running server:

```powershell
jupyter server list
```

4. Run the Playwright demo regression test with the captured URL:

```powershell
python C:\Code_win\pivotal-py\tests\jupyter_demo_test.py "<URL>"
```

5. Interpret results:

- `RESULT: PASSED`: report success and mention any warnings.
- `RESULT: FAILED`: read the error details, inspect screenshots under `tests/jupyter_screenshots/`, diagnose the problem, and report what failed.
- Exit code `2`: server connection failed; investigate the Jupyter server URL or startup state.

On first run, if no baseline exists, the script saves current outputs under `tests/jupyter_baseline/`. Report that the baseline was created. On later runs, it compares against that baseline and reports regressions such as missing charts, missing tables, or cell errors.

Passing means there are no Python tracebacks or cell errors, at least one chart/image output is present, at least one table output is present, and outputs match the saved baseline unless the baseline was just created.

## Local JupyterLab Build And Install

Use this when changing the JupyterLab extension, notebook UI behavior, syntax highlighting, autocomplete, or the Pivotal magic startup path.

From the repo root, install the core package into the active Python environment:

```powershell
python -m pip install -e .
```

On Windows, build the JupyterLab extension with the repo script:

```powershell
powershell -File C:\Code_win\pivotal-py\editors\jupyterlab\build.ps1
```

For a development build with source maps:

```powershell
powershell -File C:\Code_win\pivotal-py\editors\jupyterlab\build.ps1 -dev
```

Then install the local JupyterLab extension package:

```powershell
python -m pip install -e C:\Code_win\pivotal-py\editors\jupyterlab
jupyter labextension list
```

The extension package is `pivotal-lab`. Its JavaScript package name is `@pivotal/jupyterlab`. The build script exists because plain `jlpm build` can have Node path issues on Windows.

To launch the demo notebook locally after installing:

```powershell
powershell -File C:\Code_win\pivotal-py\start_pivotal.ps1
```

That opens `C:\Code_win\pivotal-demo\football_demo.ipynb` by default. Add `-Execute` to pre-run all cells before opening.

## Binder Demo Maintenance

The Binder demo repo is `https://github.com/nealbob/pivotal-demo.git`, usually checked out at `C:\Code_win\pivotal-demo`.

Key files in the demo repo:

- `football_demo.ipynb`: main Binder notebook.
- `football_demo.pivotal`: exported Pivotal source.
- `football_demo.py`: exported Python version of the notebook.
- `football_demo.sql`: exported SQL version.
- `binder/environment.yml`: Binder environment.
- `binder/postBuild`: installs `pivotal-lang` and the JupyterLab extension from a pinned commit of this repo.
- `binder/overrides.json`: JupyterLab settings for Binder.

When updating the Binder demo for a new Pivotal commit, update the `COMMIT=...` value in `C:\Code_win\pivotal-demo\binder\postBuild` so Binder installs the intended version from GitHub. Grammar changes often require edits to `football_demo.ipynb` and `football_demo.pivotal`, then regenerating derived artifacts.

Useful export commands from the demo repo:

```powershell
python -m pivotal --export-pivotal C:\Code_win\pivotal-demo\football_demo.ipynb
python -m pivotal --export-py C:\Code_win\pivotal-demo\football_demo.ipynb
python -m pivotal --compile --backend sql C:\Code_win\pivotal-demo\football_demo.pivotal
```

Before pushing demo repo changes, run the local Jupyter demo test from this repo and inspect any generated screenshots under `tests/jupyter_screenshots/`.

Binder rebuilds from the pushed `pivotal-demo` repository. After pushing demo changes, open the Binder link from `README.md` and confirm the notebook starts with the pinned commit from `binder/postBuild`.

If the local `C:\Code_win\pivotal-demo` checkout has unrelated edits, prefer using a clean temporary worktree from `origin/main` to make the Binder pin/update commit rather than mixing release maintenance with local notebook work.

## PyPI Release Workflow

There are two PyPI packages:

- `pivotal-lang`: core Pivotal package from the repo root `pyproject.toml`.
- `pivotal-lab`: JupyterLab extension package from `editors/jupyterlab/pyproject.toml`.

The existing release helper is `release.sh`:

```bash
./release.sh <version>
```

It updates the version in both `pyproject.toml` files, builds the root package with `python -m build`, uploads `dist/*` with `python -m twine upload dist/*`, then repeats build/upload for `editors/jupyterlab`.

On Windows, run `release.sh` from Git Bash or WSL because it uses `sed`, `rm`, and `cd` as shell commands.

Manual equivalent:

```powershell
python -m pip install --upgrade build twine
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build
python -m twine upload dist/*

Set-Location C:\Code_win\pivotal-py\editors\jupyterlab
powershell -File .\build.ps1
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build
python -m twine upload dist/*
```

Before publishing:

- Update versions in `pyproject.toml` and `editors/jupyterlab/pyproject.toml`.
- Move `CHANGELOG.md` entries from `Unreleased` into the new version section.
- Run the relevant tests, preferably `python -m pytest` for grammar/backend changes.
- Build locally and check that `dist/` contains the expected wheel and source distribution.
- Upload only the current release artifacts, not every historical file in `dist/`. Prefer explicit filenames such as `dist/pivotal_lang-0.5.0*` and `editors/jupyterlab/dist/pivotal_lab-0.5.0*`.
- Run `twine check` on both the root and JupyterLab release artifacts before upload.
- After upload, verify PyPI is serving the new versions, for example with `python -m pip index versions pivotal-lang` and `python -m pip index versions pivotal-lab`, or by installing the exact versions into a fresh temporary virtual environment.
- Use TestPyPI first if credentials or packaging changes are uncertain.

Windows notes:

- The active `python` on Windows may not see `twine`, `build`, or other user-installed scripts even after `pip install --user`. If `python -m twine` fails, use the installed executable directly from a path such as `C:\Users\<user>\AppData\Roaming\Python\Python312\Scripts\twine.exe`.
- `release.sh` is still useful, but the manual PowerShell flow is safer when you want explicit control over which artifacts are uploaded.

Recommended release order:

1. Update versions and finalize `CHANGELOG.md`.
2. Run tests and local builds.
3. Create a release commit.
4. Create an annotated tag such as `v0.5.0` from that exact release commit.
5. Push the commit and tag.
6. Publish `pivotal-lang` and `pivotal-lab`.
7. Verify installability from PyPI.

Important git note:

- Do not create the release commit and the release tag in parallel. Create the commit first, then tag that specific commit, then verify `git show v0.5.0` points at the intended release commit before publishing anything that depends on the tag.

## VS Code Extension Build And Marketplace

The VS Code extension lives in `editors/vscode`. Its package name is `pivotal-language`, publisher is `NealHughes`, and the extension version is in `editors/vscode/package.json`.

There is no repo-level VS Code marketplace release script at the moment; use `vsce` directly.

Build locally:

```powershell
Set-Location C:\Code_win\pivotal-py\editors\vscode
npm install
npm run build
```

Package a `.vsix`:

```powershell
npx @vscode/vsce package --no-dependencies --allow-missing-repository
```

Install the newest local `.vsix` for testing:

```powershell
code --install-extension .\pivotal-language-<version>.vsix --force
```

The `package.json` also has a `dev` script that builds, packages, and installs the latest `.vsix`, but it uses Unix-style shell syntax:

```bash
npm run dev
```

Use Git Bash/WSL for that script on Windows, or run the explicit PowerShell commands above.

Publish to the VS Code Marketplace with `vsce` after verifying the extension locally:

```powershell
npx @vscode/vsce publish
```

Publishing requires marketplace credentials/a Personal Access Token configured for `vsce`. If publishing a specific already-built package, use:

```powershell
npx @vscode/vsce publish --packagePath .\pivotal-language-<version>.vsix
```

Release note:

- The VS Code extension version can move independently from the Python package version. Do not assume it should exactly match `pivotal-lang`; use the next sensible extension version in `editors/vscode/package.json` and `editors/vscode/package-lock.json`.

When syntax changes, check both runtime grammar and editor grammars: `pivotal/dsl_parser.py`, `editors/vscode/syntaxes/`, `editors/jupyterlab/src/`, and the generated/built extension outputs as appropriate.

## Documentation Expectations

User-visible syntax or behavior changes should be documented close to where users will look first:

- `PIVOTAL.md` for the full syntax reference.
- `README.md` for high-level examples or headline features.
- `docs/syntax/` for topic-specific documentation.
- `docs/jupyter.md` for notebook behavior.
- `CHANGELOG.md` for release notes.

Prefer short, concrete examples over abstract descriptions.

## Online Docs Deployment

GitHub Actions automatically rebuilds and deploys the documentation when changes are pushed to the `master` branch on GitHub.

The workflow is `.github/workflows/docs.yml`. Docs are versioned with Mike:

```bash
mike deploy --push dev
```

Current docs layout:

- `stable/`: current released documentation
- `dev/`: current `master`
- `<version>/`: immutable release docs such as `0.5.0/`

The site root should redirect to `stable/`, not `dev/`.

For the first release that introduces a new stable docs version, or when repairing docs version state manually, use:

```powershell
mike deploy --push --update-aliases 0.5.0 stable
mike deploy --push dev
mike set-default -F mkdocs.yml --push stable
```

Windows notes:

- Mike may need the user scripts directory on `PATH`, for example `C:\Users\<user>\AppData\Roaming\Python\Python312\Scripts`, so that `mike`, `mkdocs`, and `ghp-import` can find each other.
- If Mike reports that the local `gh-pages` branch is unrelated to `origin/gh-pages`, recreate the local branch from the remote before deploying.
- If `mike deploy` fails while cleaning the local `site/` directory because a file is locked on Windows, remove and recreate `site/` and retry.

This means edits to `docs/`, `mkdocs.yml`, `README.md` references, or related package documentation should be committed and pushed to `master` before expecting the online docs to update.

## Release Notes

Add notable user-facing changes to `CHANGELOG.md` under `Unreleased` as the change lands. At release time, move those entries into a versioned section with the release date, then create a fresh empty `Unreleased` section.

PyPI does not automatically use `CHANGELOG.md` as the release description; PyPI uses `README.md`. If you want GitHub release notes to match the changelog, prepare a dedicated notes file such as `release-notes-0.5.0.md` from the `CHANGELOG.md` section and use it when creating the GitHub release.

Typical GitHub release command once `gh` is authenticated:

```powershell
gh release create v0.5.0 --title "Pivotal 0.5.0" --notes-file release-notes-0.5.0.md
```

## MCP Fly Deployment

The read-only hosted MCP server is deployed on Fly using `fly.toml` and `Dockerfile.mcp-readonly`.

Current hosted app:

- `pivotal-mcp-readonly`

Typical deployment flow:

```powershell
fly deploy --remote-only -a pivotal-mcp-readonly
fly logs -a pivotal-mcp-readonly --no-tail
```

Smoke-test the public endpoint:

- `https://pivotal-mcp-readonly.fly.dev/mcp`

Notes:

- The container must bind to `0.0.0.0:$PORT`. If Fly warns that the app is not listening on the expected address, check the Docker `CMD` and pass `--host 0.0.0.0 --port 8000` to `python -m pivotal.mcp_server`.
- A `406 Not Acceptable` response from `/mcp` without the right streaming `Accept` header is still useful evidence that the route is live; it is better than a timeout or `404`.
- If replacing an older Fly-hosted MCP app, either destroy the old app or scale it to zero and disable autostart so only the intended service remains active.
