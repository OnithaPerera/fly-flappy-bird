import struct
import numpy as np
import os

def generate_somas():
    points = []
    
    # region_id: 
    # 0 = Optic Lobes
    # 1 = Central Complex
    # 2 = Mushroom Body
    # 3 = Giant Fiber
    
    # 1. Central Complex (Center) - ~20k
    for _ in range(20000):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.normal(15, 3)
        y = np.random.normal(0, 2)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        points.append((x, y, z, 1.0))
        
    # 2. Optic Lobes (Left and Right) - ~50k
    for _ in range(25000):
        # Left
        x = np.random.normal(-40, 8)
        y = np.random.normal(0, 15)
        z = np.random.normal(0, 10)
        points.append((x, y, z, 0.0))
        # Right
        x = np.random.normal(40, 8)
        y = np.random.normal(0, 15)
        z = np.random.normal(0, 10)
        points.append((x, y, z, 0.0))
        
    # 3. Mushroom Body - ~40k
    for _ in range(20000):
        # Left
        x = np.random.normal(-15, 6)
        y = np.random.normal(25, 8)
        z = np.random.normal(-5, 6)
        points.append((x, y, z, 2.0))
        # Right
        x = np.random.normal(15, 6)
        y = np.random.normal(25, 8)
        z = np.random.normal(-5, 6)
        points.append((x, y, z, 2.0))
        
    # 4. Giant Fiber (Descending) - ~10k
    for _ in range(10000):
        x = np.random.normal(0, 3)
        y = np.random.normal(-30, 20)
        z = np.random.normal(0, 3)
        points.append((x, y, z, 3.0))
        
    os.makedirs("web_visualizer", exist_ok=True)
    
    with open("web_visualizer/soma_atlas.bin", "wb") as f:
        for p in points:
            f.write(struct.pack('4f', *p))
            
    print(f"Generated {len(points)} soma points in web_visualizer/soma_atlas.bin")

if __name__ == "__main__":
    generate_somas()
