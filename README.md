
# OpenCV Pose Detection Game (Proof of Concept)

This project is a proof of concept for a game that utilizes OpenCV's pose detection capabilities to control in-game actions such as jumps, direction changes, and attacks. The game demonstrates how computer vision can be integrated with simple game mechanics to create an interactive experience.

## Features

- **Pose Detection**: Uses OpenCV in conjunction with MediaPipe to detect body poses.
- **In-Game Actions**: Based on detected poses, the player can perform:
  - **Jumps**
  - **Direction Changes**
  - **Attacks**
- **Simple Game Environment**: The game runs in a basic environment created using Pygame.

## Getting Started

### Prerequisites

- Python 3.x

### Run the game:

You may need to adjust cv2.VideoCapture(0) to 1. 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pygameExample.py
```

### How to play:
lean to switch lanes, jump to jump over obstacles, kind like subway surfers. 

### How It Works

- **Pose Detection**: The game uses MediaPipe's pose estimation to track the player's body movements.
- **Mapping Poses to Actions**: Specific body movements are mapped to in-game actions. For example:
  - Raising both arms triggers a jump.
  - Leaning left or right changes the direction.
  - Punching or kicking triggers an attack.
  
- **Game Loop**: The game runs in a loop, continuously detecting poses and updating the game state accordingly.

### Next Steps

This project is a good step toward integrating pose detection with more advanced game mechanics. Future iterations will involve migrating to a game engine for more sophisticated gameplay and better performance.

## Contributing

This is a personal project, but if you have ideas or improvements, feel free to fork the repository and submit a pull request.