import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


DOCS_URL = os.environ.get("PIVOTAL_DOCS_URL", "https://docs.pivotal-lang.org/")


class LandingHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parsed_path = unquote(urlparse(path).path)
        if parsed_path.startswith("/images/"):
            local_images = Path.cwd() / "images"
            repo_images = Path.cwd().parent / "images"
            image_root = local_images if local_images.exists() else repo_images
            return str(image_root / parsed_path.removeprefix("/images/"))

        return super().translate_path(path)

    def _redirect_docs(self):
        if self.path == "/docs" or self.path.startswith("/docs/"):
            suffix = self.path[len("/docs") :].lstrip("/")
            target = DOCS_URL if not suffix else f"{DOCS_URL.rstrip('/')}/{suffix}"
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return True

        return False

    def do_GET(self):
        if self._redirect_docs():
            return

        return super().do_GET()

    def do_HEAD(self):
        if self._redirect_docs():
            return

        return super().do_HEAD()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), LandingHandler)
    server.serve_forever()
