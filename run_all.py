import subprocess
import time
import os
import http.server
import socketserver
import threading
import webbrowser

def serve_web():
    web_dir = os.path.join(os.path.dirname(__file__), "web_visualizer")
    os.chdir(web_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8080), handler) as httpd:
        print("Serving Three.js visualizer at http://localhost:8080")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start web server
    web_thread = threading.Thread(target=serve_web, daemon=True)
    web_thread.start()
    
    # Wait for server to bind
    time.sleep(1)
    
    # Open browser
    print("Launching Connectome Workbench...")
    webbrowser.open("http://localhost:8080")
    
    # Launch simulation
    print("Starting Pygame Neuro-Simulation...")
    sim = subprocess.Popen(["python", "main.py"])
    
    try:
        sim.wait()
    except KeyboardInterrupt:
        sim.terminate()
        print("Shutting down...")
