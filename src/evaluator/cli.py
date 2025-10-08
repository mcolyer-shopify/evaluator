"""CLI interface for evaluator."""

import argparse
import http.server
import json
import os
import socketserver
import sys
from pathlib import Path
from typing import Optional


def get_viewer_path() -> Path:
    """Get the path to the viewer directory."""
    return Path(__file__).parent / "viewer"


class ViewerHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for the evaluator viewer."""

    results_dir: Optional[Path] = None

    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        self.viewer_dir = get_viewer_path()
        super().__init__(*args, directory=str(self.viewer_dir), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            # Serve the main viewer page
            index_path = self.viewer_dir / "index.html"
            if not index_path.exists():
                self.send_error(404, f"index.html not found at {index_path}")
                return
            self.serve_file(index_path, "text/html")
        elif self.path.startswith("/static/"):
            # Serve static files (DuckDB WASM files)
            file_path = self.viewer_dir / self.path.lstrip("/")
            if file_path.exists() and file_path.is_file():
                # Determine content type based on extension
                if file_path.suffix == ".mjs":
                    content_type = "application/javascript"
                elif file_path.suffix == ".js":
                    content_type = "application/javascript"
                elif file_path.suffix == ".wasm":
                    content_type = "application/wasm"
                else:
                    content_type = "application/octet-stream"
                self.serve_file(file_path, content_type)
            else:
                self.send_error(404, f"File not found: {self.path}")
        elif self.path == "/api/datasets":
            # API endpoint to list available datasets
            self.serve_datasets_api()
        elif self.path.startswith("/results/"):
            # Serve result files
            filename = self.path[len("/results/") :]
            self.serve_result_file(filename)
        else:
            self.send_error(404, "Not found")

    def serve_file(self, file_path: Path, content_type: str):
        """Serve a file with the given content type."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-type", content_type)
            self.send_header("Content-Length", str(len(content)))
            # Add COOP/COEP headers for SharedArrayBuffer support
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            # Allow cross-origin access for static resources
            self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            print(f"Error serving file {file_path}: {e}", file=sys.stderr)
            self.send_error(500, f"Internal server error: {e}")

    def serve_datasets_api(self):
        """Serve the list of available datasets as JSON."""
        try:
            if not self.results_dir or not self.results_dir.exists():
                datasets = []
            else:
                datasets = []
                for file_path in self.results_dir.glob("*.duckdb"):
                    if file_path.is_file():
                        stat = file_path.stat()
                        datasets.append(
                            {
                                "name": file_path.name,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime,
                            }
                        )

                # Sort by modification time, most recent first
                datasets.sort(key=lambda x: x["mtime"], reverse=True)

            response_data = json.dumps(datasets).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(response_data)))
            # Add COOP/COEP headers
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
            self.end_headers()
            self.wfile.write(response_data)
        except Exception as e:
            print(f"Error serving datasets API: {e}", file=sys.stderr)
            self.send_error(500, f"Internal server error: {e}")

    def serve_result_file(self, filename: str):
        """Serve a result file from the results directory."""
        try:
            if not self.results_dir:
                self.send_error(404, "Results directory not configured")
                return

            # Sanitize filename to prevent directory traversal
            filename = os.path.basename(filename)
            file_path = self.results_dir / filename

            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, f"File not found: {filename}")
                return

            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            # Add COOP/COEP headers for SharedArrayBuffer support
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            print(f"Error serving result file {filename}: {e}", file=sys.stderr)
            self.send_error(500, f"Internal server error: {e}")

    def log_message(self, format, *args):
        """Override to provide cleaner log messages."""
        print(f"{self.address_string()} - {format % args}")


def viewer_command(args):
    """Start the viewer web server."""
    results_dir = Path(args.results_dir).resolve()
    viewer_path = get_viewer_path()

    # Verify viewer files exist
    if not viewer_path.exists():
        print(f"Error: Viewer directory does not exist: {viewer_path}")
        sys.exit(1)

    index_html = viewer_path / "index.html"
    if not index_html.exists():
        print(f"Error: index.html not found at: {index_html}")
        sys.exit(1)

    # Warn if results directory doesn't exist
    if not results_dir.exists():
        print(f"Warning: Results directory does not exist: {results_dir}")
        print("No datasets will be available until the directory is created.")
        print()

    # Set the results_dir as a class variable so all handlers can access it
    ViewerHTTPRequestHandler.results_dir = results_dir

    port = args.port
    host = args.host
    max_port_attempts = 10

    # Try to find an available port
    for attempt in range(max_port_attempts):
        try:
            httpd = socketserver.TCPServer((host, port), ViewerHTTPRequestHandler)
            break
        except OSError as e:
            if e.errno == 48 or "Address already in use" in str(e):
                if attempt < max_port_attempts - 1:
                    print(f"Port {port} is already in use, trying {port + 1}...")
                    port += 1
                else:
                    print(
                        f"Error: Could not find an available port after {max_port_attempts} attempts."
                    )
                    sys.exit(1)
            else:
                raise

    with httpd:
        print(f"Evaluator Viewer running at http://{host}:{port}/")
        print(f"Results directory: {results_dir}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluator - LLM Evaluation Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Viewer command
    viewer_parser = subparsers.add_parser("viewer", help="Start the results viewer")
    viewer_parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result .duckdb files (default: ./results)",
    )
    viewer_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    viewer_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    if args.command == "viewer":
        viewer_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
