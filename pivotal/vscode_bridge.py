"""
pivotal.vscode_bridge — local TCP bridge between the Python magic and the
VS Code extension.

Provides _VSCodeBridge, which implements the same send() / on_msg() interface
as ipykernel.comm.Comm so _PivotalViewer can swap it in transparently when
running inside VS Code — no changes required to any send_* method.

Protocol: newline-delimited JSON over a loopback TCP socket.  Simple, fast,
and requires no external dependencies on either side (Python socket /
Node.js net are both built-in).

Bridge file: $TMPDIR/pivotal_bridge.json  →  {"port": N, "pid": P}
Written atomically via a temp-file rename so the VS Code watcher never
reads a partial payload.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
from typing import Callable, Optional


BRIDGE_FILE = os.path.join(tempfile.gettempdir(), 'pivotal_bridge.json')


def _is_vscode() -> bool:
    """Return True if Python is running inside VS Code."""
    return (
        os.environ.get('VSCODE_PID') is not None
        or os.environ.get('TERM_PROGRAM') == 'vscode'
        or os.environ.get('VSCODE_INJECTION') == '1'
        or os.environ.get('VSCODE_CWD') is not None
        or os.environ.get('VSCODE_IPC_HOOK_CLI') is not None
    )


class _VSCodeBridge:
    """
    Local TCP bridge that relays Pivotal viewer messages to the VS Code
    extension and receives row-limit / delete messages back.

    Lifecycle
    ---------
    * Created once per IPython shell session (same as _PivotalViewer).
    * Accepts one client connection at a time; automatically re-waits after
      disconnect so the extension can reconnect after a window reload.
    * All socket I/O runs on daemon threads — the main Python thread is
      never blocked.
    """

    def __init__(self) -> None:
        self._server: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._conn_lock = threading.Lock()
        self._port: Optional[int] = None
        self._on_msg_cb: Optional[Callable] = None
        self._pending: list = []  # messages buffered while waiting for VS Code to connect
        self._start()

    # ------------------------------------------------------------------
    # Public interface — matches the subset of ipykernel.comm.Comm used
    # by _PivotalViewer
    # ------------------------------------------------------------------

    def send(self, data: dict) -> None:
        """Send a JSON message to the VS Code extension (non-blocking).

        If VS Code has not connected yet the message is buffered and flushed
        automatically when the connection is established — this handles the
        common case where Python sends data before the extension's file-watcher
        has had time to connect.
        """
        with self._conn_lock:
            conn = self._conn
        if conn is None:
            # VS Code not connected yet — buffer for flush on connect
            with self._conn_lock:
                self._pending.append(data)
            return
        self._send_to(conn, data)

    def on_msg(self, callback: Callable) -> None:
        """Register a callback for messages received from the VS Code extension.

        The callback receives a dict shaped like an ipykernel comm message::

            {'content': {'data': <payload dict>}}

        so that _PivotalViewer._on_msg() works without modification.
        """
        self._on_msg_cb = callback

    @property
    def port(self) -> Optional[int]:
        return self._port

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start(self) -> None:
        """Bind a loopback server socket on a random port and start accepting."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('127.0.0.1', 0))
        self._port = self._server.getsockname()[1]
        self._server.listen(1)
        self._write_bridge_file()
        threading.Thread(target=self._accept_loop, name='pivotal-bridge-accept',
                         daemon=True).start()

    def _write_bridge_file(self) -> None:
        """Write bridge metadata atomically so the watcher never reads partial JSON."""
        payload = json.dumps({'port': self._port, 'pid': os.getpid()})
        tmp = BRIDGE_FILE + '.tmp'
        with open(tmp, 'w') as f:
            f.write(payload)
        os.replace(tmp, BRIDGE_FILE)

    def _send_to(self, conn: socket.socket, data: dict) -> None:
        """Write one JSON line to conn; clears self._conn on error."""
        try:
            line = (json.dumps(data, default=str) + '\n').encode()
            conn.sendall(line)
        except Exception:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None

    def _accept_loop(self) -> None:
        """Accept one client at a time; re-wait after each disconnect."""
        while True:
            try:
                conn, _ = self._server.accept()
            except Exception:
                break
            with self._conn_lock:
                self._conn = conn
                pending, self._pending = list(self._pending), []
            # Flush messages that were sent before the extension connected
            for data in pending:
                self._send_to(conn, data)
                with self._conn_lock:
                    if self._conn is not conn:
                        break  # connection dropped mid-flush
            threading.Thread(target=self._read_loop, args=(conn,),
                             name='pivotal-bridge-read', daemon=True).start()

    def _read_loop(self, conn: socket.socket) -> None:
        """Read newline-delimited JSON from the extension and dispatch callbacks."""
        buf = b''
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode())
                    except Exception:
                        continue
                    cb = self._on_msg_cb
                    if cb:
                        # Wrap in the ipykernel comm envelope so _on_msg() is unchanged
                        cb({'content': {'data': msg}})
        finally:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None
