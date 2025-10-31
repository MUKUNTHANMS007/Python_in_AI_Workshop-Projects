# Gesture-Pong with Power-Ups (OpenCV + MediaPipe)
# Controls: Move hand left/right to control paddle. SPACE releases sticky ball. Q to quit.

import cv2
import mediapipe as mp
import time
import random
import math

# ---------------- Camera open helper (Windows-friendly) ----------------
def open_camera():
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        time.sleep(0.5)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return None

# ---------------- MediaPipe Hands setup ----------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- Game parameters ----------------
BASE_SPEED_MIN, BASE_SPEED_MAX = 7.0, 10.0
BALL_RADIUS = 12
PADDLE_BASE_WIDTH  = 180
PADDLE_HEIGHT      = 20
PADDLE_Y_OFFSET    = 60
HIT_SPEED_SCALE    = 1.07   # increase after each hit
MAX_SPEED          = 18.0

POWERUP_RADIUS     = 14
POWERUP_SPAWN_CHANCE = 0.25  # 25% chance on a successful hit
POWERUP_DURATION = {
    "speed": 8.0,
    "paddle": 10.0,
    "sticky": 12.0,
    "slow": 8.0
}
POWERUP_COLORS = {
    "speed":  (40, 220, 255),   # orange
    "paddle": (120, 255, 120),  # green
    "sticky": (220, 120, 255),  # purple
    "slow":   (255, 120, 120),  # blue-ish/red
}
EFFECT_BADGE_COLORS = {
    "speed":  (80, 170, 255),
    "paddle": (100, 200, 100),
    "sticky": (200, 120, 255),
    "slow":   (255, 140, 140),
}

def reset_ball(w, h):
    vx = random.choice([-1.0, 1.0]) * random.uniform(BASE_SPEED_MIN, BASE_SPEED_MAX)
    vy = -random.uniform(BASE_SPEED_MIN, BASE_SPEED_MAX)
    return [w//2, h//2], [vx, vy]

def clamp_speed(vx, vy, max_speed=MAX_SPEED):
    speed = math.hypot(vx, vy)
    if speed > max_speed:
        scale = max_speed / (speed + 1e-6)
        return vx*scale, vy*scale
    return vx, vy

def spawn_powerup(w, h):
    ptype = random.choice(list(POWERUP_DURATION.keys()))
    x = random.randint(80, w-80)
    y = random.randint(120, h//2)  # spawn in upper half
    return {"type": ptype, "pos": [x, y], "active": False, "taken": False}

def draw_powerup(frame, p):
    color = POWERUP_COLORS.get(p["type"], (255,255,255))
    cv2.circle(frame, (int(p["pos"][0]), int(p["pos"][1])), POWERUP_RADIUS, color, -1)
    cv2.putText(frame, p["type"], (int(p["pos"][0]-28), int(p["pos"][1]-20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    cap = open_camera()
    if cap is None:
        raise RuntimeError("Could not open camera. Check privacy settings, close apps using camera, and ensure non-headless OpenCV is installed.")

    # Game state
    score = 0
    misses = 0
    combo  = 0
    last_hit_time = 0.0

    # Activated effects with expiry
    active_effects = {}  # name -> expiry_time
    sticky_armed = False  # if True, the next paddle contact sticks the ball

    # PowerUp on field
    field_powerup = None

    # Initial frame for dimensions
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read from camera.")
    h, w = frame.shape[:2]

    ball_pos, ball_vel = reset_ball(w, h)
    paddle_x = w // 2

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Hand tracking -> paddle position
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            hand_x = int(lm[5].x * w)  # index MCP x for stability
            paddle_x = hand_x

        # Remove expired effects
        now = time.time()
        for k in list(active_effects.keys()):
            if now >= active_effects[k]:
                del active_effects[k]

        # Paddle width with effect
        paddle_width = PADDLE_BASE_WIDTH
        if "paddle" in active_effects:
            paddle_width = int(PADDLE_BASE_WIDTH * 1.4)

        # Paddle bounds and rect
        paddle_x = max(paddle_width//2, min(w - paddle_width//2, paddle_x))
        paddle_y = h - PADDLE_Y_OFFSET
        paddle_left  = paddle_x - paddle_width//2
        paddle_right = paddle_x + paddle_width//2
        paddle_top   = paddle_y - PADDLE_HEIGHT//2
        paddle_bot   = paddle_y + PADDLE_HEIGHT//2

        # Input
        key = cv2.waitKey(1) & 0xFF

        # Sticky behavior: lock to paddle until SPACE pressed
        if "sticky" in active_effects and sticky_armed:
            ball_pos[0] = paddle_x
            ball_pos[1] = paddle_top - BALL_RADIUS - 1
            cv2.putText(frame, "Sticky active: press SPACE to launch", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,120,255), 2)
            if key == ord(' '):
                vx = random.choice([-1.0, 1.0]) * random.uniform(BASE_SPEED_MIN, BASE_SPEED_MAX)
                vy = -random.uniform(BASE_SPEED_MIN, BASE_SPEED_MAX)
                ball_vel = [vx, vy]
                ball_vel[0], ball_vel[1] = clamp_speed(ball_vel[0], ball_vel[1], MAX_SPEED)
                sticky_armed = False
        else:
            # Update ball physics
            ball_pos[0] += ball_vel[0]
            ball_pos[1] += ball_vel[1]

        # Wall collisions
        if ball_pos[0] <= BALL_RADIUS:
            ball_pos[0] = BALL_RADIUS
            ball_vel[0] *= -1
        if ball_pos[0] >= w - BALL_RADIUS:
            ball_pos[0] = w - BALL_RADIUS
            ball_vel[0] *= -1
        if ball_pos[1] <= BALL_RADIUS:
            ball_pos[1] = BALL_RADIUS
            ball_vel[1] *= -1

        # Paddle collision if not sticky-locked
        if not ("sticky" in active_effects and sticky_armed):
            if (paddle_left - BALL_RADIUS <= ball_pos[0] <= paddle_right + BALL_RADIUS) and \
               (paddle_top  - BALL_RADIUS <= ball_pos[1] <= paddle_bot   + BALL_RADIUS) and \
               (time.time() - last_hit_time > 0.1):
                # Position above paddle and bounce up
                ball_pos[1] = paddle_top - BALL_RADIUS
                ball_vel[1] = -abs(ball_vel[1])
                # English based on contact point
                offset = (ball_pos[0] - paddle_x) / (paddle_width/2)
                ball_vel[0] += 5.0 * offset

                # Progressive difficulty + active effects
                speed_scale = HIT_SPEED_SCALE
                if "speed" in active_effects:
                    speed_scale *= 1.3
                if "slow" in active_effects:
                    speed_scale *= 0.8
                ball_vel[0] *= speed_scale
                ball_vel[1] *= speed_scale
                ball_vel[0], ball_vel[1] = clamp_speed(ball_vel[0], ball_vel[1], MAX_SPEED)

                score += 1
                combo += 1
                last_hit_time = time.time()

                # Spawn a power-up sometimes
                if field_powerup is None and random.random() < POWERUP_SPAWN_CHANCE:
                    field_powerup = spawn_powerup(w, h)

                # Arm sticky after a successful hit if active
                if "sticky" in active_effects:
                    sticky_armed = True

        # Miss (ball below screen)
        if ball_pos[1] >= h + BALL_RADIUS:
            misses += 1
            combo = 0
            ball_pos, ball_vel = reset_ball(w, h)
            sticky_armed = False

        # Power-up collection by ball
        if field_powerup and not field_powerup["taken"]:
            dx = ball_pos[0] - field_powerup["pos"][0]
            dy = ball_pos[1] - field_powerup["pos"][1]
            if math.hypot(dx, dy) <= (BALL_RADIUS + POWERUP_RADIUS):
                ptype = field_powerup["type"]
                active_effects[ptype] = time.time() + POWERUP_DURATION.get(ptype, 8.0)
                if ptype == "speed":
                    ball_vel[0] *= 1.3
                    ball_vel[1] *= 1.3
                if ptype == "slow":
                    ball_vel[0] *= 0.75
                    ball_vel[1] *= 0.75
                if ptype == "sticky":
                    sticky_armed = True
                ball_vel[0], ball_vel[1] = clamp_speed(ball_vel[0], ball_vel[1], MAX_SPEED)
                field_powerup["taken"] = True

        # Remove taken power-up after a short linger
        if field_powerup and field_powerup["taken"]:
            field_powerup = None

        # Draw paddle and ball
        cv2.rectangle(frame, (paddle_left, paddle_top), (paddle_right, paddle_bot), (60, 220, 60), -1)
        cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS, (220, 220, 60), -1)

        # Draw power-up if present
        if field_powerup and not field_powerup["taken"]:
            draw_powerup(frame, field_powerup)

        # Draw hand landmarks for feedback
        if res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # HUD
        cv2.rectangle(frame, (0,0), (w, 90), (30,30,30), -1)
        cv2.putText(frame, f"Gesture-Pong+ | Score: {score}  Misses: {misses}  Combo: x{combo}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Active effects with timers
        x0 = 20
        for eff in ["speed", "paddle", "sticky", "slow"]:
            if eff in active_effects:
                tleft = max(0.0, active_effects[eff] - time.time())
                label = f"{eff}:{int(tleft)}s"
                color = EFFECT_BADGE_COLORS.get(eff, (200,200,200))
                cv2.putText(frame, label, (x0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                x0 += 160

        cv2.putText(frame, "Move hand to control paddle | SPACE: release sticky | Q: quit",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,60,200), 2)

        cv2.imshow("Gesture-Pong+ (Power-Ups)", frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
