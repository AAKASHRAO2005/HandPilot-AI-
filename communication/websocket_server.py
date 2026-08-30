"""
communication/websocket_server.py — Local WebSocket server for optional browser extension.

Runs in a background thread using asyncio. Broadcasts gesture events to all
connected browser extension clients as JSON messages.

Binds ONLY to 127.0.0.1 — not exposed to the network.
"""

import asyncio
import json
import threading
from typing import Optional, Set

from utils.logger import setup_logger

log = setup_logger(__name__)


class WebSocketServer:
    """
    Async WebSocket server broadcasting gesture events on localhost.

    Args:
        host: Bind address (always 127.0.0.1).
        port: TCP port (default 8765).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._clients: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._server = None

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="WebSocketServer"
        )
        self._thread.start()
        log.info(f"WebSocket server starting on ws://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        log.info("WebSocket server stopped.")

    def broadcast(self, event: dict) -> None:
        """
        Thread-safe broadcast of a gesture event dict to all connected clients.

        Args:
            event: Gesture event as a dict (will be JSON-serialised).
        """
        if self._loop and self._running and self._clients:
            asyncio.run_coroutine_threadsafe(self._async_broadcast(event), self._loop)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            log.error(f"WebSocket server error: {e}")
        finally:
            # Cancel all pending tasks cleanly before closing the loop
            try:
                pending = asyncio.all_tasks(self._loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            self._loop.close()

    async def _serve(self) -> None:
        try:
            import websockets
            async with websockets.serve(
                self._handler,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
            ) as server:
                self._server = server
                log.info(f"WebSocket server ready on ws://{self.host}:{self.port}")
                while self._running:
                    await asyncio.sleep(0.1)
        except Exception as e:
            log.error(f"WebSocket serve error: {e}")

    async def _handler(self, websocket, path=None) -> None:
        self._clients.add(websocket)
        log.info(f"Browser extension connected. Total clients: {len(self._clients)}")
        try:
            await websocket.send(json.dumps({"type": "connected", "message": "Hand Gesture Controller"}))
            async for message in websocket:
                # Echo / handle incoming commands from extension
                try:
                    data = json.loads(message)
                    log.debug(f"Extension message: {data}")
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            log.info(f"Browser extension disconnected. Total clients: {len(self._clients)}")

    async def _async_broadcast(self, event: dict) -> None:
        if not self._clients:
            return
        msg = json.dumps(event)
        dead = set()
        for ws in self._clients.copy():
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead
