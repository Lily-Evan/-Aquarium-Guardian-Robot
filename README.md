
# 🐠 Aquarium Guardian Robot

> **Aquarium Guardian** is a Python simulation of an autonomous robot designed to maintain a virtual fish tank.  
> The robot interacts with fish, algae, and the aquarium environment, performing tasks such as feeding, cleaning, and health monitoring.

---

## ✨ Features

🐟 Fish Agents
• Schooling, random motion & wall avoidance
• Stress levels with color feedback
• Healthy = green | Stressed = reddish

🌊 Aquarium Environment
• Algae growth on glass
• Turbidity & temperature cycles
• Food particles falling & sinking

🤖 Guardian Robot
• Patrols around the tank
• Feeds fish (scheduled or opportunistic)
• Cleans algae using its brush
• Calms stressed fish (health check)
• Priority: fish health > feeding > cleaning > patrol

🖥 Visualization
• OpenCV grid-based rendering
• On-screen HUD with stats
• Exports aquarium_guardian.mp4

yaml
Αντιγραφή κώδικα

---

## 🛠 Requirements

```bash
Python 3.9+
pip install numpy opencv-python
🚀 Usage
bash
Αντιγραφή κώδικα
# Run the simulation
python aquarium_guardian.py

# Controls
ESC → Exit the simulation window

# Output
aquarium_guardian.mp4 is saved automatically
📊 Configurable Parameters
python
Αντιγραφή κώδικα
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
Αντιγραφή κώδικα
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
