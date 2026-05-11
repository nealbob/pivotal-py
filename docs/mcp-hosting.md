# MCP Server

Pivotal includes a hosted-safe MCP mode for ChatGPT, Claude, and other remote
MCP clients. This mode exposes only documentation and compile-time tools:

- `pivotal_syntax`
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

## Local Smoke Test

Install the MCP extra:

```powershell
python -m pip install -e ".[mcp]"
```

Start the read-only HTTP server:

```powershell
python -m pivotal.mcp_server --read-only --transport streamable-http --host 0.0.0.0 --port 8000
```

The MCP endpoint is:

```text
http://localhost:8000/mcp
```

The existing local stdio server is unchanged:

```powershell
python -m pivotal.mcp_server
```

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
- Start without authentication only for early testing; add rate limiting or auth
  before advertising it broadly.
- `pivotal_compile` validates parsing/code generation only. It does not prove
  semantic equivalence to pandas because it does not run data.
