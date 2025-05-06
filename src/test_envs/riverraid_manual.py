import gymnasium as gym
import pygame
import numpy as np
import time
from src.lib import wrappers

# Initialize pygame for capturing keyboard input
pygame.init()
pygame.display.set_caption("Riverraid Manual Play")
screen = pygame.display.set_mode((500, 600))
clock = pygame.time.Clock()

# Create the environment
ENV_NAME = "RiverraidNoFrameskip-v4"
env = gym.make(ENV_NAME, render_mode="rgb_array")  # Using raw env without wrappers for rendering

# Define key mappings (Riverraid has 18 actions)
KEY_ACTION_MAP = {
    pygame.K_UP: 2,      # UP - Fire
    pygame.K_RIGHT: 8,   # RIGHT - Move right
    pygame.K_LEFT: 4,    # LEFT - Move left
    pygame.K_DOWN: 5,    # DOWN - Do nothing/slow down
    pygame.K_SPACE: 1,   # SPACE - Alternative fire
}

def display_frame(frame):
    """Convert the game frame to a pygame surface and display it"""
    # Resize frame to fit our window
    frame = np.transpose(frame, (1, 0, 2))  # Transpose for pygame's display order
    surf = pygame.surfarray.make_surface(frame)
    surf = pygame.transform.scale(surf, (500, 600))
    screen.blit(surf, (0, 0))
    
    # Add game info as text
    font = pygame.font.Font(None, 36)
    info_text = f"Score: {total_reward:.1f} | Press ESC to quit"
    text_surf = font.render(info_text, True, (255, 255, 255))
    screen.blit(text_surf, (10, 10))
    
    pygame.display.flip()

# Reset the environment
observation, info = env.reset()
frame = env.render()
total_reward = 0
done = False
action = 0  # Default action (NOOP)

print("Controls:")
print("  ↑ (UP) - Fire")
print("  → (RIGHT) - Move right")
print("  ← (LEFT) - Move left")
print("  ↓ (DOWN) - Do nothing/slow down")
print("  SPACE - Alternative fire")
print("  ESC - Quit")

# Main game loop
running = True
while running:
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key in KEY_ACTION_MAP:
                action = KEY_ACTION_MAP[event.key]
        elif event.type == pygame.KEYUP:
            if event.key in KEY_ACTION_MAP:
                action = 0  # Reset to NOOP when key is released
    
    # Process keys being held down
    keys = pygame.key.get_pressed()
    for key, act in KEY_ACTION_MAP.items():
        if keys[key]:
            action = act
            break
    
    # Take action in the environment
    observation, reward, done, truncated, info = env.step(action)
    frame = env.render()
    total_reward += reward
    
    # Display the game
    display_frame(frame)
    
    # Check if episode is done
    if done or truncated:
        print(f"Episode finished! Score: {total_reward}")
        observation, info = env.reset()
        total_reward = 0
    
    # Control game speed
    clock.tick(30)  # 30 FPS

# Clean up
env.close()
pygame.quit()