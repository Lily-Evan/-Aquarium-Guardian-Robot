# aquarium_guardian.py
# Aquarium Guardian Robot — simulation prototype
# deps: numpy, opencv-python

import numpy as np
import cv2
import random
from math import hypot, atan2, cos, sin

# -----------------------------
# Parameters
# -----------------------------
H, W = 90, 160         # grid size (cells)
SCALE = 6              # upscaling for display
STEPS = 1500
FPS = 24

N_FISH = 12
FISH_SPEED = 0.6       # cells / step
SCHOOLING_WEIGHT = 0.25
WALL_REPULSION = 1.0

ALGAE_GROWTH = 0.0015  # per step on glass
CLEAN_RADIUS = 2       # robot cleaning brush radius
FEED_INTERVAL = 350    # steps between scheduled feedings
FEED_PORT = (3, W//2)  # feeder location (near top center)

ROBOT_SPEED = 1.0
ROBOT_ACCEL = 0.4

# Colors (BGR)
COL_WATER = (232, 238, 246)
COL_GLASS = (80, 80, 80)
COL_ALGAE = (60, 150, 60)
COL_ROBOT = (70, 70, 255)
COL_FOOD  = (40, 170, 255)

rng = np.random.default_rng(7)

# -----------------------------
# Utility helpers
# -----------------------------
def clamp(x, a, b): return a if x < a else b if x > b else x
def within(x, y): return 1 <= x < H-1 and 1 <= y < W-1

# -----------------------------
# Fish agent
# -----------------------------
class Fish:
    def __init__(self):
        self.x = rng.uniform(10, H-10)
        self.y = rng.uniform(10, W-10)
        theta = rng.uniform(0, 2*np.pi)
        self.vx = FISH_SPEED * np.cos(theta)
        self.vy = FISH_SPEED * np.sin(theta)
        self.stress = 0.0   # 0..1
        self.color = (30, 180, 30)  # greener = healthy; paler with stress

    def step(self, fishes):
        # Cohesion (schooling): move toward local centroid
        neighbors = [f for f in fishes if f is not self and (abs(f.x-self.x)+abs(f.y-self.y)) < 20]
        dvx = dvy = 0.0
        if neighbors:
            cx = sum(f.x for f in neighbors) / len(neighbors)
            cy = sum(f.y for f in neighbors) / len(neighbors)
            ang = atan2(cx - self.x, cy - self.y)
            dvx += SCHOOLING_WEIGHT * cos(ang)
            dvy += SCHOOLING_WEIGHT * sin(ang)

        # Wall repulsion
        repx = (1.0/(self.x-1.5) - 1.0/(H-2.5-self.x))
        repy = (1.0/(self.y-1.5) - 1.0/(W-2.5-self.y))
        dvx += WALL_REPULSION * repx * 0.02
        dvy += WALL_REPULSION * repy * 0.02

        # Random jitter
        dvx += rng.normal(0, 0.05)
        dvy += rng.normal(0, 0.05)

        # integrate velocity & clamp speed
        self.vx = 0.9*self.vx + dvx
        self.vy = 0.9*self.vy + dvy
        speed = hypot(self.vx, self.vy) + 1e-9
        if speed > FISH_SPEED:
            self.vx *= FISH_SPEED / speed
            self.vy *= FISH_SPEED / speed

        self.x += self.vx
        self.y += self.vy
        self.x = clamp(self.x, 1.5, H-2.5)
        self.y = clamp(self.y, 1.5, W-2.5)

        # Stress heuristic: low motion for long → stress increases slightly
        mot = speed
        self.stress = clamp(0.98*self.stress + (0.05 if mot < 0.15 else -0.03), 0.0, 1.0)
        # Color: pale with stress
        g = int(clamp(180 - 100*self.stress, 60, 200))
        r = int(clamp(30 + 40*self.stress, 30, 120))
        self.color = (30, g, r)  # BGR-ish greenish fish

# -----------------------------
# Aquarium world
# -----------------------------
class Aquarium:
    def __init__(self):
        # glass mask: the 1-cell border inside the tank
        self.glass = np.zeros((H, W), np.uint8)
        self.glass[1, 1:W-1] = 1
        self.glass[H-2, 1:W-1] = 1
        self.glass[1:H-1, 1] = 1
        self.glass[1:H-1, W-2] = 1

        self.algae = np.zeros((H, W), np.float32)  # 0..1 on glass only
        self.turbidity = 0.1
        self.temperature = 24.0  # C

        # food particles (y descends slowly)
        self.food = []  # list of (x, y, ttl)

    def step_env(self, fishes, step):
        # algae grows on glass; more with high turbidity & sunlight hours
        sunlight = 1.0 if (8*FPS) < step % (24*FPS) < (18*FPS) else 0.6
        growth = ALGAE_GROWTH * sunlight * (0.8 + 0.4*rng.random())
        self.algae[self.glass == 1] = np.clip(self.algae[self.glass == 1] + growth, 0, 1)

        # turbidity increases with algae + fish activity; decreases slowly (filter)
        avg_speed = np.mean([hypot(f.vx, f.vy) for f in fishes]) if fishes else 0.0
        self.turbidity = 0.98*self.turbidity + 0.02*(0.3*self.algae.mean() + 0.7*avg_speed)

        # temperature small daily cycle
        dayphase = (step % (24*FPS)) / (24*FPS)
        self.temperature = 24.0 + 1.0*np.sin(2*np.pi*dayphase)

        # update food particles
        new_food = []
        for x, y, ttl in self.food:
            ny = y + 0.25  # sink
            nt = ttl - 1
            if nt > 0 and ny < H-2:
                new_food.append((x, ny, nt))
        self.food = new_food

    def add_food(self, amount=40):
        # spawn near feeder port (small spread)
        for _ in range(amount):
            x = FEED_PORT[0] + rng.normal(0, 0.5)
            y = FEED_PORT[1] + rng.normal(0, 3.0)
            self.food.append((clamp(x, 2, H-3), clamp(y, 2, W-3), rng.integers(200, 400)))

    def clean_algae_disk(self, cx, cy, radius):
        xs = np.arange(max(1, int(cx-radius)), min(H-1, int(cx+radius+1)))
        ys = np.arange(max(1, int(cy-radius)), min(W-1, int(cy+radius+1)))
        for i in xs:
            for j in ys:
                if self.glass[i, j] == 1 and (i-cx)**2 + (j-cy)**2 <= radius**2:
                    self.algae[i, j] = 0.0

# -----------------------------
# Robot
# -----------------------------
class Robot:
    def __init__(self, x=H-5, y=5):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.task = "patrol"
        self.target = None
        self.brush_on = False
        # patrol waypoints (on glass)
        self.waypoints = [(2, 2), (2, W-3), (H-3, W-3), (H-3, 2)]
        self.wp_idx = 0
        self.feed_cooldown = 0

    def set_task(self, task, target=None):
        self.task = task
        self.target = target

    def goto(self, tx, ty):
        # simple PD-like chase
        dx, dy = tx - self.x, ty - self.y
        dist = max(1e-6, hypot(dx, dy))
        ax, ay = (dx / dist) * ROBOT_ACCEL, (dy / dist) * ROBOT_ACCEL
        self.vx = 0.85*self.vx + ax
        self.vy = 0.85*self.vy + ay
        spd = hypot(self.vx, self.vy)
        if spd > ROBOT_SPEED:
            self.vx *= ROBOT_SPEED / spd
            self.vy *= ROBOT_SPEED / spd
        self.x += self.vx
        self.y += self.vy
        # arrived?
        return dist < 1.0

    def step(self, aq: Aquarium, fishes, step):
        self.feed_cooldown = max(0, self.feed_cooldown - 1)
        self.brush_on = False

        # --- task arbitration ---
        # 1) feeding: scheduled or opportunistic (fish cluster near surface)
        time_to_feed = (step % FEED_INTERVAL == 0)
        cluster_near_surface = sum(1 for f in fishes if f.x < 8) >= max(4, N_FISH//3)
        need_feed = (time_to_feed or cluster_near_surface) and self.feed_cooldown == 0

        # 2) algae cleaning: look for max algae on glass
        alg_mat = aq.algae * aq.glass
        (ai, aj) = np.unravel_index(np.argmax(alg_mat), alg_mat.shape)
        high_algae = alg_mat[ai, aj] > 0.35

        # 3) fish health check: any highly stressed fish?
        stressed = [f for f in fishes if f.stress > 0.65]
        need_check = len(stressed) > 0

        # Priority: emergency (fish) > feeding > cleaning > patrol
        if need_check:
            tgt = min(stressed, key=lambda f: hypot(f.x-self.x, f.y-self.y))
            self.set_task("check_fish", (tgt.x, tgt.y))
        elif need_feed:
            self.set_task("feed", FEED_PORT)
        elif high_algae:
            self.set_task("clean", (ai, aj))
        elif self.task not in ("patrol",):
            self.set_task("patrol")

        # --- execute task ---
        if self.task == "feed" and self.target:
            arrived = self.goto(*self.target)
            if arrived:
                aq.add_food(amount=60)
                self.feed_cooldown = FEED_INTERVAL//2
                self.set_task("patrol")
        elif self.task == "clean" and self.target:
            arrived = self.goto(*self.target)
            if arrived:
                aq.clean_algae_disk(int(self.x), int(self.y), CLEAN_RADIUS)
                self.brush_on = True
                # continue cleaning along glass line locally
                self.set_task("patrol")
        elif self.task == "check_fish" and self.target:
            arrived = self.goto(*self.target)
            if arrived:
                # “observe” fish: slight calming effect
                for f in fishes:
                    if hypot(f.x - self.x, f.y - self.y) < 6:
                        f.stress *= 0.9
                self.set_task("patrol")
        else:
            # patrol waypoints
            wx, wy = self.waypoints[self.wp_idx]
            arrived = self.goto(wx, wy)
            if arrived:
                self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)

# -----------------------------
# Rendering
# -----------------------------
def render(aq: Aquarium, fishes, robot: Robot, step, rec_title="Aquarium Guardian"):
    canvas = np.full((H, W, 3), COL_WATER, np.uint8)

    # draw glass
    glass_coords = np.where(aq.glass == 1)
    canvas[glass_coords] = COL_GLASS

    # draw algae as overlay on glass
    if aq.algae.max() > 0:
        a = (aq.algae * 255).astype(np.uint8)
        algae_rgb = np.zeros_like(canvas)
        algae_rgb[..., 0] = COL_ALGAE[0]
        algae_rgb[..., 1] = (a * 0.7).astype(np.uint8)
        algae_rgb[..., 2] = COL_ALGAE[2]
        mask = aq.algae > 0.01
        canvas[mask] = (0.5*canvas[mask] + 0.5*algae_rgb[mask]).astype(np.uint8)

    # draw food
    for x, y, ttl in aq.food:
        xi, yi = int(x), int(y)
        if within(xi, yi):
            canvas[xi, yi] = COL_FOOD

    # draw fish
    for f in fishes:
        xi, yi = int(f.x), int(f.y)
        if within(xi, yi):
            # fish body
            canvas[xi, yi] = f.color
            # small tail indication
            tx, ty = int(clamp(xi - np.sign(f.vx), 1, H-2)), int(clamp(yi - np.sign(f.vy), 1, W-2))
            canvas[tx, ty] = (max(f.color[0]-20,0), max(f.color[1]-20,0), max(f.color[2]-20,0))

    # draw robot
    rx, ry = int(robot.x), int(robot.y)
    cv2.circle(canvas, (ry, rx), 2, COL_ROBOT, -1)
    if robot.brush_on:
        cv2.circle(canvas, (ry, rx), CLEAN_RADIUS, (255, 180, 60), 1)

    # upscale
    vis = cv2.resize(canvas, (W*SCALE, H*SCALE), interpolation=cv2.INTER_NEAREST)

    # HUD
    hud1 = f"Step {step} | Turbidity {aq.turbidity:.2f} | Temp {aq.temperature:.1f}C"
    hud2 = f"Task: {robot.task} | Fish stressed: {sum(1 for f in fishes if f.stress>0.65)} | Food: {len(aq.food)}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(vis, hud1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(vis, hud2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(vis, rec_title, (vis.shape[1]-360, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1, cv2.LINE_AA)
    return vis

# -----------------------------
# Main loop
# -----------------------------
def main():
    aq = Aquarium()
    fishes = [Fish() for _ in range(N_FISH)]
    robot = Robot()

    first = render(aq, fishes, robot, 0)
    h, w, _ = first.shape
    out = cv2.VideoWriter("aquarium_guardian.mp4", cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))

    for step in range(1, STEPS+1):
        # environment
        aq.step_env(fishes, step)

        # fish step
        for f in fishes:
            f.step(fishes)

        # opportunistic feeding behavior: fish swim upward if food present near surface
        if aq.food:
            for f in fishes:
                # nudge toward nearest food
                fx, fy, _ = min(aq.food, key=lambda p: hypot(f.x - p[0], f.y - p[1]))
                ang = atan2(fx - f.x, fy - f.y)
                f.vx = 0.8*f.vx + 0.2*cos(ang)*FISH_SPEED
                f.vy = 0.8*f.vy + 0.2*sin(ang)*FISH_SPEED

        # robot step
        robot.step(aq, fishes, step)

        # render
        frame = render(aq, fishes, robot, step)
        cv2.imshow("Aquarium Guardian Robot", frame)
        out.write(frame)
        if cv2.waitKey(10) == 27:  # ESC
            break

    out.release()
    cv2.destroyAllWindows()
    print("Done. Video saved as aquarium_guardian.mp4")

if __name__ == "__main__":
    main()
