import http.server
import socketserver
import threading
import time
import webbrowser

import click

from tsundoku.utils.config import TsundokuApp


@click.command("open_report")
@click.argument("experiment", type=str)
@click.argument("port", type=int, default=8000)
@click.argument("host", type=str, default="localhost")
@click.argument("delay", type=int, default=3)
def main(experiment, port, host, delay):
    app = TsundokuApp("Report Viewer")
    report_path = app.project_path / "reports" / experiment

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(report_path), **kwargs)

    def serve_forever():
        with socketserver.TCPServer((host, port), Handler) as httpd:
            httpd.serve_forever()

    thread = threading.Thread(target=serve_forever)
    thread.daemon = False
    thread.start()
    time.sleep(delay)
    webbrowser.open_new_tab(f"http://{host}:{port}/index.html")


if __name__ == "__main__":
    main()
