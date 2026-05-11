# MCP Server

Pivotal includes an optional MCP server for connecting Pivotal syntax, docs,
compilation, execution, and comparison tools to MCP-capable AI clients.

There are two main ways to run it:

| Mode | Best for | Tools exposed |
|------|----------|---------------|
| Local full server | Local development, private data, pandas-to-Pivotal conversion checks | Docs, syntax, examples, compile, highlight, run, compare |
| Hosted read-only server | Public or remote clients where execution should be disabled | Docs, syntax, examples, compile, highlight |

Install the MCP extra before using either mode:

```powershell
python -m pip install "pivotal-lang[mcp]"
```

For an editable checkout:

```powershell
python -m pip install -e ".[mcp]"
```

## Local Full Server

Use the local full server when the MCP client is running on your machine and you
want it to execute Pivotal pipelines, load local CSV/Parquet inputs, or compare
pandas code with Pivotal code.

Start the default stdio server:

```powershell
python -m pivotal.mcp_server
```

This mode exposes:

- `pivotal_syntax`
- `pivotal_docs_index`
- `pivotal_docs`
- `pivotal_docs_search`
- `pivotal_examples`
- `pivotal_compile`
- `pivotal_highlight`
- `pivotal_run`
- `pivotal_compare`
- `pivotal_compare_files`

`pivotal_run` can execute Pivotal source in an isolated subprocess and return
structured results, generated code, warnings, and table previews.
`pivotal_compare` and `pivotal_compare_files` compare pandas and Pivotal outputs
on the same inputs.

When passing data files to run or compare tools, use a mapping from table name to
local CSV or Parquet path:

```json
{
  "sales": "C:\\data\\sales.csv"
}
```

The key is the table or variable name made available to the tool, not the file
name.

## Local HTTP Server

For a local client that expects an HTTP MCP endpoint, run the full server with an
HTTP transport:

```powershell
python -m pivotal.mcp_server --transport streamable-http --host 127.0.0.1 --port 8000
```

The endpoint is:

```text
http://127.0.0.1:8000/mcp
```

Keep the full server bound to `127.0.0.1` unless you specifically intend to
expose execution tools beyond your machine.

## Hosted Read-Only Server

Use read-only mode for hosted or public MCP clients. This mode is designed for
learning, documentation lookup, syntax highlighting, and compile checks without
executing Python or reading user data files.

Start a read-only Streamable HTTP server:

```powershell
python -m pivotal.mcp_server --read-only --transport streamable-http --host 0.0.0.0 --port 8000
```

The MCP endpoint is:

```text
http://localhost:8000/mcp
```

Read-only mode exposes:

- `pivotal_syntax`
- `pivotal_docs_index`
- `pivotal_docs`
- `pivotal_docs_search`
- `pivotal_examples`
- `pivotal_compile`
- `pivotal_highlight`

It does not expose `pivotal_run`, `pivotal_compare`, or file-based comparison
tools, so remote users can parse and compile Pivotal code but cannot execute
Python/data pipelines or read files from the server.

`pivotal_highlight` returns copyable HTML by default: the highlighted code is
wrapped in a styled `<pre>` block with a top-right button that copies the plain
Pivotal source via `innerText`. Pass `include_copy_button=false` when a client
needs only the bare highlighted span fragment.

## Docker

Build the read-only image:

```powershell
docker build -f Dockerfile.mcp-readonly -t pivotal-mcp-readonly .
```

Run it locally:

```powershell
docker run --rm -p 8000:8000 pivotal-mcp-readonly
```

The endpoint is:

```text
http://localhost:8000/mcp
```

## Deploying

Deploy `Dockerfile.mcp-readonly` to a container host such as Fly.io, Railway, or
Render. The container listens on `$PORT` and serves Streamable HTTP MCP at:

```text
https://YOUR_HOST/mcp
```

Use that URL as the custom MCP connector URL in a remote client.

## Notes

- Use HTTPS for public clients.
- Start hosted servers without authentication only for early testing; add rate
  limiting or auth before advertising them broadly.
- Do not expose the full local server publicly unless you explicitly want remote
  clients to execute code or read paths that the server process can access.
- `pivotal_compile` validates parsing/code generation only. It does not prove
  semantic equivalence to pandas because it does not run data.
