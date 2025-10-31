
# AI Hand Gesture Projects (Python)

This repository contains three AI-based hand gesture control projects developed during an AI Workshop.  
Each project uses computer vision techniques to detect and interpret hand gestures in real-time using a webcam.

---

## Projects Included

### 1. Stone Paper Scissor
A classic Stone–Paper–Scissors game controlled by hand gestures.

- Detects your gesture (rock, paper, or scissors) using OpenCV and MediaPipe.  
- The computer randomly selects its move.  
- Displays the winner on the screen.

**File:** `Stone_paper_scissor.py`

---

### 2. Subway Surfers Hand Gesture Controller
Control the Subway Surfers game using hand gestures.

- Tracks your hand movement via the webcam.  
- Maps gestures to game controls such as jump, duck, and move left/right.  
- Provides a fun and interactive way to play without using the keyboard.

**File:** `Subway Surfers Hand Gesture Controller.py`

---

### 3. Ping Pong (Hand Gesture Controlled)
A virtual Ping Pong game where your hand acts as the paddle.

- Uses real-time hand tracking to move the paddle.  
- The ball bounces dynamically based on paddle interaction.  
- Demonstrates interactive vision-based control using hand gestures.

**File:** `ping_pong.py`

---

## Technologies Used
- Python 3  
- OpenCV  
- MediaPipe (for hand detection)  
- NumPy  
- PyAutoGUI  

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>

2. Install the required libraries:

   ```bash
   pip install opencv-python mediapipe numpy pyautogui
   ```

3. Run any of the Python files:

   ```bash
   python Stone_paper_scissor.py
   ```

4. Allow camera access and perform gestures in front of the webcam.

---

## Acknowledgment

These projects were developed as part of an AI Workshop to explore the use of hand gesture recognition in interactive applications.



