# Rock–Paper–Scissors with Computer Vision (OpenCV + MediaPipe)
# Press SPACE to start a round, hold your gesture during the countdown, Q to quit.

import cv2
import mediapipe as mp
import random
import time
from collections import deque

# ---------------- Camera open helper (Windows-friendly) ----------------
def open_camera():
    # Try common indices with DirectShow backend (more stable on Windows)
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        time.sleep(0.5)  # warmup
        if cap.isOpened():
            # Optional: set resolution for consistency
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return None  # Let caller raise a clear error

# ---------------- Game / CV setup ----------------
CHOICES = ["rock", "paper", "scissors"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# A small buffer to smooth predictions near resolution time
last_moves = deque(maxlen=8)

def get_finger_states(lm):
    """
    Return list of 5 ints indicating if finger is 'up' (1) or 'down' (0).
    Order: [thumb, index, middle, ring, pinky]
    Uses y-coordinates for fingers (except thumb uses x due to lateral motion).
    """
    # Determine left/right hand roughly using index vs pinky MCP x
    index_mcp_x = lm[5].x
    pinky_mcp_x = lm[17].x
    right_hand = index_mcp_x < pinky_mcp_x  # heuristic

    fingers = [0, 0, 0, 0, 0]

    # Thumb: compare x, direction flips by hand orientation
    if right_hand:
        fingers[0] = 1 if lm[4].x < lm[3].x else 0
    else:
        fingers[0] = 1 if lm[4].x > lm[3].x else 0

    # Other fingers: tip above pip -> up (y smaller is higher)
    fingers[1] = 1 if lm[8].y  < lm[6].y  else 0   # index
    fingers[2] = 1 if lm[12].y < lm[10].y else 0   # middle
    fingers[3] = 1 if lm[16].y < lm[14].y else 0   # ring
    fingers[4] = 1 if lm[20].y < lm[18].y else 0   # pinky
    return fingers

def classify_gesture(fingers):
    """
    Convert finger states into rock/paper/scissors/none with relaxed thresholds:
    - rock: <=1 finger up (allow slight noise)
    - paper: >=4 fingers up (allow thumb variance)
    - scissors: index and middle up, ring and pinky down (thumb free)
    """
    up = sum(fingers)
    if up <= 1:
        return "rock"
    if up >= 4:
        return "paper"
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "scissors"
    return "none"

def decide_winner(player, comp):
    if player == comp:
        return "draw"
    if (player == "rock" and comp == "scissors") or \
       (player == "paper" and comp == "rock") or \
       (player == "scissors" and comp == "paper"):
        return "player"
    return "computer"

def draw_hud(frame, countdown, player_move, comp_move, result, score_p, score_c, round_active):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w, 100), (30,30,30), -1)
    cv2.putText(frame, f"Countdown: {countdown}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"Player: {player_move}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)
    cv2.putText(frame, f"Comp: {comp_move}", (260, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,255), 2)
    cv2.putText(frame, f"Result: {result}", (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,220,180), 2)

    msg = "Show your move now!" if round_active else "Press SPACE to start a round"
    color = (60,200,60) if round_active else (60,60,200)
    cv2.putText(frame, msg, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.rectangle(frame, (0,h-60), (w,h), (30,30,30), -1)
    cv2.putText(frame, f"Score  You {score_p}  -  {score_c} CPU", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

def main():
    cap = open_camera()
    if cap is None:
        raise RuntimeError("Could not open camera. Check privacy settings, close apps using camera, and ensure non-headless OpenCV is installed.")

    score_player, score_comp = 0, 0
    round_active = False
    round_start_time = 0.0
    player_move, comp_move, result = "none", "none", "..."
    countdown_val = 3

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # Draw hand landmarks for feedback
        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            round_active = True
            round_start_time = time.time()
            player_move, comp_move, result = "none", "none", "..."
            countdown_val = 3
            last_moves.clear()
        if key == ord('q'):
            break

        if round_active:
            elapsed = time.time() - round_start_time
            # Show integer countdown
            remaining = max(0, 3 - int(elapsed))
            countdown_val = remaining if remaining > 0 else 0

            # Sample moves between 2.0s and 3.2s, then resolve at >=3.2s
            if 2.0 <= elapsed < 3.2 and res.multi_hand_landmarks:
                lm = list(res.multi_hand_landmarks[0].landmark)
                fingers = get_finger_states(lm)
                last_moves.append(classify_gesture(fingers))

            if elapsed >= 3.2:
                if len(last_moves):
                    move_counts = {m: list(last_moves).count(m) for m in ["rock","paper","scissors","none"]}
                    player_move = max(move_counts, key=move_counts.get)
                else:
                    player_move = "none"

                comp_move = random.choice(CHOICES)
                if player_move == "none":
                    result = "no move"
                else:
                    winner = decide_winner(player_move, comp_move)
                    if winner == "draw":
                        result = "draw"
                    elif winner == "player":
                        result = "you win"
                        score_player += 1
                    else:
                        result = "cpu wins"
                        score_comp += 1

                round_active = False
                last_moves.clear()

        draw_hud(frame, countdown_val, player_move, comp_move, result, score_player, score_comp, round_active)
        cv2.putText(frame, "Q to quit", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,50), 2)
        cv2.imshow("Rock-Paper-Scissors CV", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
