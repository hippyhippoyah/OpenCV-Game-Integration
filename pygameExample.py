import pygame
import random
import multiprocessing
from pose_detection import run_pose_detection

# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 60
GROUND_LEVEL = SCREEN_HEIGHT - PLAYER_HEIGHT - 10
LANE_WIDTH = SCREEN_WIDTH // 3
OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 60
OBSTACLE_SPEED = 5
FRAME_RATE = 60
LEAN_COOLDOWN_FRAMES = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

def pygame_app(queue):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # Player variables
    player_lane = 1  # Start in the middle lane (0: left, 1: middle, 2: right)
    player_y = GROUND_LEVEL
    jump = False
    is_jumping = False
    jump_height = 100
    jump_speed = 15
    attack = False

    # Obstacles
    obstacles = []
    obstacle_spawn_timer = 0
    obstacle_spawn_interval = 60  # Spawn an obstacle every 2 seconds
    lean_cooldown = 0

    lean = None
    jump = False
    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Check if there is data in the queue
        if not queue.empty():
            data = queue.get()
            jump = data.get("jump", False)
            lean = data.get("lean", None)

        # Handle jump
        if jump and not is_jumping:
            is_jumping = True
            jump_speed = 15

        if is_jumping:
            player_y -= jump_speed
            jump_speed -= 1
            if player_y >= GROUND_LEVEL:
                player_y = GROUND_LEVEL
                is_jumping = False

        # Handle lean
        if lean_cooldown == 0:
            if lean == "left":
                player_lane -= 1
            elif lean == "right":
                player_lane += 1
            lean_cooldown = LEAN_COOLDOWN_FRAMES
        else:
            lean_cooldown -= 1
        

        # Keep player within the lanes
        player_lane = max(0, min(2, player_lane))

        # Spawn obstacles
        obstacle_spawn_timer += 1
        if obstacle_spawn_timer >= obstacle_spawn_interval:
            obstacle_lane = random.randint(0, 2)
            obstacle_y = -OBSTACLE_HEIGHT
            obstacles.append([obstacle_lane, obstacle_y])
            obstacle_spawn_timer = 0

        # Move obstacles
        for obstacle in obstacles:
            obstacle[1] += OBSTACLE_SPEED

        # Remove off-screen obstacles
        obstacles = [obstacle for obstacle in obstacles if obstacle[1] < SCREEN_HEIGHT]

        # Check for collisions
        player_x = player_lane * LANE_WIDTH + (LANE_WIDTH - PLAYER_WIDTH) // 2
        for obstacle in obstacles:
            obstacle_x = obstacle[0] * LANE_WIDTH + (LANE_WIDTH - OBSTACLE_WIDTH) // 2
            obstacle_y = obstacle[1]
            if (player_x < obstacle_x + OBSTACLE_WIDTH and
                player_x + PLAYER_WIDTH > obstacle_x and
                player_y < obstacle_y + OBSTACLE_HEIGHT and
                player_y + PLAYER_HEIGHT > obstacle_y):
                print("Collision! Game Over.")
                # pygame.quit()
                # return

        # Drawing
        screen.fill(BLACK)

        # Draw lanes
        pygame.draw.line(screen, WHITE, (LANE_WIDTH, 0), (LANE_WIDTH, SCREEN_HEIGHT), 5)
        pygame.draw.line(screen, WHITE, (2 * LANE_WIDTH, 0), (2 * LANE_WIDTH, SCREEN_HEIGHT), 5)

        # Draw player
        pygame.draw.rect(screen, BLUE, (player_x, player_y, PLAYER_WIDTH, PLAYER_HEIGHT))

        # Draw obstacles
        for obstacle in obstacles:
            obstacle_x = obstacle[0] * LANE_WIDTH + (LANE_WIDTH - OBSTACLE_WIDTH) // 2
            obstacle_y = obstacle[1]
            pygame.draw.rect(screen, RED, (obstacle_x, obstacle_y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

        pygame.display.flip()

        # Control the frame rate
        clock.tick(FRAME_RATE)

if __name__ == '__main__':
    # Create a queue for inter-process communication
    queue = multiprocessing.Queue()

    # Start the pose detection process
    p1 = multiprocessing.Process(target=run_pose_detection, args=(queue,))
    p1.start()

    # Start the Pygame application
    pygame_app(queue)

    # Wait for the pose detection process to finish
    p1.join()
