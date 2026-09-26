import asyncio
import websockets
import json
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TelemetryServer")

class TelemetryServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.thread = None

    async def _handler(self, websocket, path=None):
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        try:
            # We don't expect messages from client, but we must read to keep connection alive
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # websockets >= 10.0 handles 'path' differently or drops it, but the default serve behavior is fine
        start_server = websockets.serve(self._handler, self.host, self.port)
        
        self.loop.run_until_complete(start_server)
        logger.info(f"Telemetry WebSocket server listening on ws://{self.host}:{self.port}")
        self.loop.run_forever()

    def start(self):
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def broadcast(self, data: dict):
        if not self.loop or not self.clients:
            return
            
        message = json.dumps(data)
        
        async def _broadcast():
            disconnected = set()
            for ws in self.clients:
                try:
                    await ws.send(message)
                except Exception:
                    disconnected.add(ws)
            for ws in disconnected:
                self.clients.remove(ws)
                
        asyncio.run_coroutine_threadsafe(_broadcast(), self.loop)

server = TelemetryServer()
