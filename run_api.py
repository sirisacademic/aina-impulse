
import argparse
import socket
import uvicorn
from src.impulse.settings import settings

def find_free_port(preferred: int = 8000, limit: int = 20, host: str = "127.0.0.1") -> int:
    candidates = [preferred] + list(range(preferred + 1, preferred + 1 + limit))
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=settings.api_host, help="Host to bind")
    p.add_argument("--port", type=int, default=int(settings.api_port), help="Preferred port (0=auto)")
    p.add_argument("--scan-limit", type=int, default=20, help="How many ports to try after preferred")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    host = args.host
    if args.port == 0:
        port = 0
    else:
        probe_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
        port = find_free_port(preferred=args.port, limit=args.scan_limit, host=probe_host)

    uvicorn.run(
        "src.impulse.api.main:app",
        host=host,
        port=port,
        reload=True,
    )
