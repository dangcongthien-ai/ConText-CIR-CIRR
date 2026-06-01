from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerUrls:
    bind_url: str
    local_url: str
    lan_url: str | None


def _detect_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            lan_ip = sock.getsockname()[0]
    except OSError:
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            return None
    return None if lan_ip.startswith("127.") else lan_ip


def resolve_server_urls(host: str, port: int) -> ServerUrls:
    bind_url = f"http://{host}:{port}"
    local_url = f"http://127.0.0.1:{port}"
    lan_url = None
    if host in {"0.0.0.0", "::"}:
        lan_ip = _detect_lan_ip()
        if lan_ip:
            lan_url = f"http://{lan_ip}:{port}"
    return ServerUrls(bind_url=bind_url, local_url=local_url, lan_url=lan_url)
