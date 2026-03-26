"""
Playwright-based smoke test for the Pivotal JupyterLab demo notebook.

Assumes JupyterLab is already running. Pass the full URL (with token) as
the first argument, or set JUPYTER_URL env var.

Usage:
    python tests/jupyter_demo_test.py "http://127.0.0.1:8888/lab?token=abc123"
    python tests/jupyter_demo_test.py "http://..." --backend polars

Exit codes:
    0  — all checks passed
    1  — one or more checks failed
    2  — could not connect / notebook not found
"""
import sys
import os
import json
import time
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

NOTEBOOK_PATH = "football_demo.ipynb"
TIMEOUT = 120_000   # ms — cells can be slow on first run
CELL_TIMEOUT = 360_000  # 6 minutes — football demo loads a large dataset

# Backend-specific patterns to verify in code output
BACKEND_CODE_PATTERNS = {
    'polars': ['import polars as pl', 'pl.col(', 'pl.DataFrame', 'pl.read_'],
    'duckdb': ['_pivotal_ddb', 'SELECT', 'duckdb'],
    'pandas': ['pd.DataFrame', '.query(', '.merge(', 'pd.read_'],
}


def log(msg):
    print(msg, flush=True)


def run(jupyter_url: str, backend: str = 'pandas'):
    base_dir = Path(__file__).parent
    screenshot_dir = base_dir / f"jupyter_screenshots/{backend}"
    baseline_dir   = base_dir / f"jupyter_baseline/{backend}"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 900})
        page = context.new_page()

        # ----------------------------------------------------------------
        # 1. Navigate to the notebook
        # ----------------------------------------------------------------
        # Build notebook URL from lab URL
        base = re.sub(r'\?.*$', '', jupyter_url)  # strip query string
        base = re.sub(r'/lab/?$', '', base)
        token_match = re.search(r'token=([a-zA-Z0-9]+)', jupyter_url)
        token_param = f"?token={token_match.group(1)}" if token_match else ""
        notebook_url = f"{base}/lab/tree/{NOTEBOOK_PATH}{token_param}"

        log(f"Navigating to: {notebook_url}")
        try:
            page.goto(notebook_url, timeout=TIMEOUT)
        except PWTimeout:
            log("ERROR: Could not reach JupyterLab — is the server running?")
            browser.close()
            return 2

        # ----------------------------------------------------------------
        # 2. Wait for JupyterLab to fully load
        # ----------------------------------------------------------------
        log("Waiting for JupyterLab to load...")
        try:
            # The notebook tab title appears once the file is open
            page.wait_for_selector(".jp-NotebookPanel", timeout=TIMEOUT)
            log("JupyterLab loaded.")
        except PWTimeout:
            page.screenshot(path=str(screenshot_dir / "load_failure.png"))
            log("ERROR: JupyterLab did not load in time. Screenshot saved.")
            browser.close()
            return 2

        # ----------------------------------------------------------------
        # 3. Inject backend override cell (if not pandas)
        # ----------------------------------------------------------------
        if backend != 'pandas':
            log(f"Injecting backend override: {backend}")
            _insert_backend_cell(page, backend)

        # ----------------------------------------------------------------
        # 4. Restart kernel and run all cells
        # ----------------------------------------------------------------
        log("Restarting kernel and running all cells...")
        # Use the Kernel menu
        page.click('text=Kernel')
        page.wait_for_selector('.lm-Menu', timeout=10_000)
        page.click('text=Restart Kernel and Run All Cells…')

        # Confirm the dialog
        try:
            page.wait_for_selector('button:has-text("Restart")', timeout=10_000)
            page.click('button:has-text("Restart")')
            log("Kernel restarted, cells running...")
        except PWTimeout:
            log("WARNING: Restart confirmation dialog not found — may have auto-confirmed.")

        # ----------------------------------------------------------------
        # 5. Wait for all cells to finish
        # ----------------------------------------------------------------
        log("Waiting for all cells to complete...")
        try:
            # Step 1: wait for execution to start (at least one [*]: appears).
            # Without this, the check below can fire immediately in the gap
            # between kernel restart and when cells begin running.
            page.wait_for_function(
                """() => Array.from(
                    document.querySelectorAll('.jp-InputArea-prompt')
                ).some(el => el.textContent.trim() === '[*]:')""",
                timeout=30_000,
                polling=500,
            )
            log("Execution started, waiting for completion...")
            # Step 2: wait for all [*]: to disappear
            page.wait_for_function(
                """() => {
                    const prompts = document.querySelectorAll('.jp-InputArea-prompt');
                    if (prompts.length === 0) return false;
                    return !Array.from(prompts).some(
                        el => el.textContent.trim() === '[*]:'
                    );
                }""",
                timeout=CELL_TIMEOUT,
                polling=2000,
            )
            log("All cells completed.")
        except PWTimeout:
            page.screenshot(path=str(screenshot_dir / "timeout.png"))
            log("ERROR: Cells did not finish within timeout. Screenshot saved.")
            browser.close()
            return 1

        # Allow charts/widgets to finish rendering
        time.sleep(3)

        # ----------------------------------------------------------------
        # 6. Check generated code patterns (backend-specific)
        # ----------------------------------------------------------------
        patterns = BACKEND_CODE_PATTERNS.get(backend, [])
        if patterns:
            log(f"Checking code output for {backend} patterns...")
            all_output_text = _collect_all_output_text(page)
            found_patterns = [p for p in patterns if p in all_output_text]
            missing_patterns = [p for p in patterns if p not in all_output_text]
            if found_patterns:
                log(f"PASS: Found {backend} code patterns: {found_patterns}")
            if missing_patterns:
                log(f"WARN: Missing {backend} code patterns: {missing_patterns}")

        # ----------------------------------------------------------------
        # 7. Take screenshots
        # ----------------------------------------------------------------
        log("Taking screenshots...")
        # Scroll to top first so full_page captures from the beginning
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        page.screenshot(path=str(screenshot_dir / "notebook_top.png"))
        page.screenshot(path=str(screenshot_dir / "notebook_full.png"), full_page=True)
        # Also screenshot the right sidebar (Pivotal viewer panel)
        sidebar = page.query_selector('#jp-right-stack')
        if sidebar and sidebar.is_visible():
            sidebar.screenshot(path=str(screenshot_dir / "viewer_panel.png"))
            log("Viewer panel screenshot saved.")
        elif sidebar:
            log("INFO: Right sidebar exists but is not visible (may be collapsed).")
        log(f"Screenshots saved to {screenshot_dir}")

        # ----------------------------------------------------------------
        # 6. Check for errors in cell outputs
        # ----------------------------------------------------------------
        log("Checking for errors in cell outputs...")
        error_cells = page.query_selector_all('.jp-RenderedText[data-mime-type="application/vnd.jupyter.stderr"]')
        traceback_cells = page.query_selector_all('.jp-OutputArea-output.jp-RenderedText')

        errors_found = []
        for el in traceback_cells:
            text = el.inner_text()
            if 'Error' in text or 'Traceback' in text:
                errors_found.append(text[:200])

        if errors_found:
            log(f"FAIL: {len(errors_found)} cell(s) have errors:")
            for e in errors_found:
                log(f"  - {e}")
        else:
            log("PASS: No errors in cell outputs.")

        # ----------------------------------------------------------------
        # 7. Check expected outputs exist
        # ----------------------------------------------------------------
        log("Checking expected outputs...")
        # Pivotal renders charts/tables into the right sidebar viewer panel,
        # not as inline cell outputs — check both locations
        # Pivotal renders into the left sidebar object panel and right viewer panel
        checks = [
            ('.jp-OutputArea img',            "Inline chart/image outputs"),
            ('#jp-right-stack img',           "Viewer panel chart/image"),
            ('#jp-right-stack canvas',        "Viewer panel canvas"),
        ]
        # Pivotal object panel items (left sidebar)
        pivotal_items = page.query_selector_all('.pivotal-object-item, [class*="pivotal"]')

        all_passed = not errors_found
        viewer_has_content = False
        for selector, description in checks:
            found = page.query_selector_all(selector)
            if found:
                log(f"PASS: {description} ({len(found)} found)")
                viewer_has_content = True
            else:
                log(f"INFO: {description} — none found")

        if pivotal_items:
            log(f"PASS: Pivotal object panel has {len(pivotal_items)} item(s)")
            viewer_has_content = True

        if not viewer_has_content:
            log("WARN: Pivotal viewer panel appears empty — charts/tables may not have rendered")

        # ----------------------------------------------------------------
        # 8. Cycle through viewer outputs and screenshot each one
        # ----------------------------------------------------------------
        log("Cycling through viewer outputs...")
        viewer_screenshots = _screenshot_viewer_outputs(page, screenshot_dir)
        log(f"Captured {len(viewer_screenshots)} viewer output screenshot(s).")

        # ----------------------------------------------------------------
        # 9. Compare against baseline if it exists
        # ----------------------------------------------------------------
        baseline_file = baseline_dir / "cell_outputs.json"
        if baseline_file.exists():
            log("Comparing against baseline...")
            with open(baseline_file) as f:
                baseline = json.load(f)
            current_outputs = _collect_outputs(page)
            mismatches = _compare_outputs(baseline, current_outputs)
            if mismatches:
                log(f"FAIL: {len(mismatches)} output(s) differ from baseline:")
                for m in mismatches:
                    log(f"  Cell {m['cell']}: {m['issue']}")
                all_passed = False
            else:
                log(f"PASS: All {len(baseline)} baseline outputs match.")
        else:
            log("No baseline found — saving current outputs as baseline.")
            baseline_dir.mkdir(exist_ok=True)
            outputs = _collect_outputs(page)
            with open(baseline_file, 'w') as f:
                json.dump(outputs, f, indent=2)
            # Also save a baseline screenshot
            page.screenshot(
                path=str(baseline_dir / "notebook_baseline.png"),
                full_page=True
            )
            log(f"Baseline saved to {baseline_dir}")

        browser.close()

        if all_passed:
            log("\nRESULT: PASSED")
            return 0
        else:
            log("\nRESULT: FAILED")
            return 1


def _insert_backend_cell(page, backend: str):
    """Insert a %pivotal_set backend=<backend> cell at the top of the notebook."""
    # Click the first cell to select it
    first_cell = page.query_selector('.jp-Cell')
    if not first_cell:
        log("  WARNING: no cells found to insert above")
        return
    first_cell.click()
    time.sleep(0.3)

    # Enter command mode, go to first cell, insert above
    page.keyboard.press('Escape')
    page.keyboard.press('Control+Home')
    time.sleep(0.2)
    page.keyboard.press('a')  # insert cell above in command mode
    time.sleep(0.3)

    # Enter edit mode and type the magic command
    page.keyboard.press('Enter')
    page.keyboard.type(f'%pivotal_set backend={backend}')
    page.keyboard.press('Escape')
    time.sleep(0.2)
    log(f"  Inserted cell: %pivotal_set backend={backend}")


def _collect_all_output_text(page) -> str:
    """Collect all text from cell output areas (for code pattern checking)."""
    outputs = page.query_selector_all('.jp-OutputArea-output')
    text_parts = []
    for el in outputs:
        text_parts.append(el.inner_text())
    return '\n'.join(text_parts)


def _screenshot_viewer_outputs(page, screenshot_dir) -> list:
    """
    Click each chart and table item in the Pivotal explorer panel, wait for the
    viewer to render, and save a screenshot of the right viewer panel.

    The Pivotal explorer uses:
      .pv-explorer-folder-header  — section headers (DATA, CHARTS, TABLES)
      .pv-explorer-row            — individual item rows
      .pv-explorer-name           — the name span inside each row

    Returns list of saved file paths.
    """
    saved = []
    sidebar = page.query_selector('#jp-right-stack')

    # Collect all explorer rows and the section header that preceded each.
    # We only want rows under CHARTS or TABLES sections.
    all_rows = page.query_selector_all('.pv-explorer-row, .pv-explorer-folder-header')
    current_section = ''
    clickable = []   # (section, name, element)
    for el in all_rows:
        cls = el.get_attribute('class') or ''
        if 'pv-explorer-folder-header' in cls:
            label_el = el.query_selector('.pv-explorer-folder-label')
            current_section = label_el.inner_text().strip() if label_el else ''
        elif 'pv-explorer-row' in cls and current_section in ('CHARTS', 'TABLES'):
            name_el = el.query_selector('.pv-explorer-name')
            name = name_el.inner_text().strip() if name_el else ''
            if name:
                clickable.append((current_section, name, el))

    if not clickable:
        log("  No chart/table rows found in explorer — skipping viewer screenshots.")
        return saved

    # Collect names only — re-query elements before each click since the
    # explorer re-renders after each selection, making stored handles stale.
    items_to_click = [(section, name) for section, name, _ in clickable]

    for section, name in items_to_click:
        log(f"  Opening {section}: {name}")
        # Re-query the specific row by matching its name text
        rows = page.query_selector_all('.pv-explorer-row')
        target = None
        for row in rows:
            name_el = row.query_selector('.pv-explorer-name')
            if name_el and name_el.inner_text().strip() == name:
                target = row
                break
        if not target:
            log(f"  WARNING: could not find row for '{name}' — skipping")
            continue
        target.click()
        time.sleep(1.2)  # wait for viewer to render
        if sidebar:
            safe_name = re.sub(r'[^\w]', '_', name)
            path = str(screenshot_dir / f"viewer_{safe_name}.png")
            sidebar.screenshot(path=path)
            saved.append(path)
            log(f"  Saved viewer_{safe_name}.png")

    return saved


def _collect_outputs(page) -> list:
    """Collect text content of all cell outputs for baseline comparison."""
    outputs = []
    cells = page.query_selector_all('.jp-Cell')
    for i, cell in enumerate(cells):
        output_area = cell.query_selector('.jp-OutputArea')
        if output_area:
            # Collect text outputs
            text_out = output_area.inner_text().strip()
            # Note whether there are images/tables (don't compare pixel-exact)
            has_image = bool(output_area.query_selector('img'))
            has_table = bool(output_area.query_selector('table'))
            if text_out or has_image or has_table:
                outputs.append({
                    'cell': i,
                    'has_image': has_image,
                    'has_table': has_table,
                    'text_snippet': text_out[:300],
                })
    return outputs


def _compare_outputs(baseline: list, current: list) -> list:
    """Compare current outputs against baseline. Returns list of mismatches."""
    mismatches = []

    # Check same number of cells with output
    if len(baseline) != len(current):
        mismatches.append({
            'cell': 'summary',
            'issue': f"Output cell count changed: baseline={len(baseline)}, current={len(current)}"
        })

    for b, c in zip(baseline, current):
        if b['has_image'] and not c['has_image']:
            mismatches.append({'cell': b['cell'], 'issue': "Image output missing"})
        if b['has_table'] and not c['has_table']:
            mismatches.append({'cell': b['cell'], 'issue': "Table output missing"})

    return mismatches


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('url', nargs='?', default=os.environ.get('JUPYTER_URL', ''),
                        help='JupyterLab URL with token')
    parser.add_argument('--backend', default='pandas',
                        choices=['pandas', 'polars', 'duckdb'],
                        help='Pivotal backend to test (default: pandas)')
    args = parser.parse_args()

    if not args.url:
        print("Usage: python tests/jupyter_demo_test.py <jupyter-url-with-token> [--backend polars]")
        sys.exit(2)
    sys.exit(run(args.url, backend=args.backend))
