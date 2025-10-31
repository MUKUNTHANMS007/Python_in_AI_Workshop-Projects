
import cv2
import mediapipe as mp
import time
import math
import pyautogui

# ------------- Input sender (swap to pyKey if needed) -------------
def send_key(key_name):
    # Default: pyautogui
    pyautogui.press(key_name)

# Alternative (if game ignores pyautogui on Windows):
# from pyKey import pressKey, releaseKey, W, A, S, D, UP, DOWN, LEFT, RIGHT
# KEY_MAP = {"left": LEFT, "right": RIGHT, "up": UP, "down": DOWN}
# def send_key(key_name):
#     vk = KEY_MAP[key_name]
#     pressKey(vk); time.sleep(0.02); releaseKey(vk)

# ------------- MediaPipe Hands -------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ------------- Gesture detection params -------------
HIST_LEN = 6                 # frames for velocity calc
SWIPE_COOLDOWN = 0.45        # seconds
JUMP_COOLDOWN  = 0.60
ROLL_COOLDOWN  = 0.60
VEL_X_THRESH   = 0.25        # normalized px/frame for swipe
VEL_Y_THRESH   = 0.25        # upward velocity threshold for jump
FIST_UP_THRESH = 2           # fingers up count threshold for fist (<=2)
OPEN_UP_THRESH = 4           # open palm if >= 4 fingers up

# Track landmark history (wrist pos)
pos_hist = []   # list of (t, x_norm, y_norm)
last_swipe_t = 0.0
last_jump_t  = 0.0
last_roll_t  = 0.0

def fingers_up(lm):
    # Returns count of fingers up based on tip vs pip y (thumb uses x).
    # Indices: Thumb(4 vs 3), Index(8 vs 6), Middle(12 vs 10),
    # Ring(16 vs 14), Pinky(20 vs 18)
    # Determine right-hand heuristic
    right_hand = lm[5].x < lm[17].x
    up = 0
    # Thumb
    if right_hand:
        up += 1 if lm[4].x < lm[3].x else 0
    else:
        up += 1 if lm[4].x > lm[3].x else 0
    # Others
    up += 1 if lm[8].y  < lm[6].y  else 0
    up += 1 if lm[12].y < lm[10].y else 0
    up += 1 if lm[16].y < lm[14].y else 0
    up += 1 if lm[20].y < lm[18].y else 0
    return up

def detect_swipe_and_actions(frame_w, frame_h, lm, now):
    global pos_hist, last_swipe_t, last_jump_t, last_roll_t

    # Normalize wrist position
    wx = lm[0].x  # wrist
    wy = lm[0].y

    pos_hist.append((now, wx, wy))
    if len(pos_hist) > HIST_LEN:
        pos_hist.pop(0)

    # Need at least 2 points
    if len(pos_hist) < 2:
        return None

    # Velocity: average between first and last in history (per frame normalized units)
    t0, x0, y0 = pos_hist[0]
    t1, x1, y1 = pos_hist[-1]
    dt = max(1e-3, t1 - t0)
    vx = (x1 - x0) / dt
    vy = (y1 - y0) / dt

    # Count fingers
    fup = fingers_up(lm)

    # Swipe Left/Right (cooldown)
    if now - last_swipe_t >= SWIPE_COOLDOWN:
        if vx <= -VEL_X_THRESH:
            send_key("left")
            last_swipe_t = now
            return "left"
        elif vx >= VEL_X_THRESH:
            send_key("right")
            last_swipe_t = now
            return "right"

    # Jump: fast upward motion OR open palm moving up (cooldown)
    if now - last_jump_t >= JUMP_COOLDOWN:
        # vy negative means moving up in normalized coords
        if (-vy) >= VEL_Y_THRESH or (fup >= OPEN_UP_THRESH and (-vy) >= VEL_Y_THRESH*0.6):
            send_key("up")
            last_jump_t = now
            return "up"

    # Roll: fist pose (<= 2 fingers up) and relatively steady hand (cooldown)
    if now - last_roll_t >= ROLL_COOLDOWN:
        if fup <= FIST_UP_THRESH:
            send_key("down")
            last_roll_t = now
            return "down"

    return None

def main():
    # Speed up pyautogui to avoid delays
    try:
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
    except Exception:
        pass

    # Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # fallback indices
        for idx in [1,2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                break
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    action_text = "Ready"
    action_t    = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            # Draw landmarks
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            # Detect actions
            act = detect_swipe_and_actions(w, h, lm, now)
            if act:
                action_text = f"Action: {act}"
                action_t = now
        else:
            # Decay history to avoid stale velocity
            if len(pos_hist) > 0:
                pos_hist.pop(0)

        # HUD
        cv2.rectangle(frame, (0,0), (w, 70), (30,30,30), -1)
        cv2.putText(frame, "Subway Surfers Gesture Controller", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Show action briefly
        if now - action_t < 0.6:
            cv2.putText(frame, action_text, (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60,220,220), 2)
        else:
            cv2.putText(frame, "Gestures: Swipe L/R, Up to Jump, Fist to Roll",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120,120,255), 2)

        cv2.putText(frame, "Tip: Focus the game window so keys register. Q to quit.",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

        cv2.imshow("Gesture -> Keys (Subway Surfers)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
