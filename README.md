# 🐠 Aquarium Guardian Robot

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green?logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Aquarium Guardian** is a Python simulation of an autonomous robot designed to maintain a virtual fish tank.  
The robot interacts with fish, algae, and the aquarium environment, performing tasks such as feeding, cleaning, and health monitoring.

---

## ✨ Features

### 🐟 Fish Agents
- Schooling, random motion & wall avoidance  
- Stress levels with color feedback  
- Healthy = green | Stressed = reddish  

### 🌊 Aquarium Environment
- Algae growth on glass  
- Turbidity & temperature cycles  
- Food particles falling & sinking  

### 🤖 Guardian Robot
- Patrols around the tank  
- Feeds fish (scheduled or opportunistic)  
- Cleans algae using its brush  
- Calms stressed fish (health check)  
- **Task priority:** Fish health > Feeding > Cleaning > Patrol  

### 🖥 Visualization
- OpenCV grid-based rendering  
- On-screen HUD with stats  
- Exports `aquarium_guardian.mp4`  

---

🚀 Usage
bash

# Run the simulation
python aquarium_guardian.py

# Controls
ESC → Exit the simulation window

# Output
aquarium_guardian.mp4 is saved automatically
📊 Configurable Parameters
python

N_FISH        # number of fish
FISH_SPEED    # swimming speed
ALGAE_GROWTH  # algae growth rate
FEED_INTERVAL # feeding frequency (steps)
ROBOT_SPEED   # robot max speed
ROBOT_ACCEL   # robot acceleration
📹 Example Output
Fish swimming, schooling & changing color with stress

Robot patrolling, feeding, cleaning algae

Dynamic algae growth & turbidity changes

📐 System Architecture
lua

+------------+        +--------------+        +--------+
|   Fish     | -----> | Environment  | -----> | Render |
+------------+        +--------------+        +--------+
       ^                      |
       |                      v
       +-----------------+  Robot  |
                         +---------+
💡 Future Ideas
🤖 Machine Learning for adaptive cleaning/feeding

🕹 GUI for manual user control

📊 Export stats to CSV

🔌 Integration with real hardware (microcontrollers)

## 🛠 Requirements

```bash
Python 3.9+
pip install numpy opencv-python




