import gymnasium as gym
import pygame
import numpy as np
import time
from src.lib import wrappers
import collections

# Initialize pygame for capturing keyboard input
pygame.init()
pygame.display.set_caption("Riverraid Manual Play")
screen = pygame.display.set_mode((1000, 1500))  # Increased width and height for larger display
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

# RAM values to monitor - we'll display all RAM values from 0-255
NUM_RAM_BYTES = 128  # Atari 2600 has 128 bytes of RAM
last_ram_values = {}

# For tracking fuel level
CANDIDATE_FUEL_ADDRESSES = [120]  # Address 109 is the fuel level
CANDIDATE_COLORS = [(0, 255, 0)]  # Green

# For tracking history
FUEL_HISTORY_SIZE = 200
fuel_history = {addr: collections.deque(maxlen=FUEL_HISTORY_SIZE) for addr in CANDIDATE_FUEL_ADDRESSES}
fuel_collection_events = []
MAX_EVENTS = 10
frame_counter = 0

def get_ram_values():
    """Get all RAM values from the Atari environment"""
    ram = env.unwrapped.ale.getRAM()
    return {i: ram[i] for i in range(NUM_RAM_BYTES)}

def draw_ram_values(screen, values, last_values, highlight_address=120):
    """Draw RAM values in a clear, fixed layout"""
    # Clear the display area first
    pygame.draw.rect(screen, (0, 0, 0), (0, 490, 1000, 410))
    
    # Use a fixed-width font for better alignment
    font = pygame.font.SysFont('courier', 18)  # Slightly larger font
    
    # Title
    title_font = pygame.font.SysFont('arial', 24, bold=True)  # Larger title font
    title = title_font.render("RAM Values (Address: Value | Change)", True, (255, 255, 255))
    screen.blit(title, (10, 495))
    
    # Number of columns and rows
    num_cols = 4
    num_rows = 32
    col_width = 240  # Wider columns
    row_height = 22  # Taller rows
    start_y = 530  # Adjusted start_y to leave room for title
    
    # Draw background for the RAM values area
    pygame.draw.rect(screen, (20, 20, 30), (5, start_y-5, col_width*num_cols+10, row_height*num_rows+15), 0)
    pygame.draw.rect(screen, (100, 100, 120), (5, start_y-5, col_width*num_cols+10, row_height*num_rows+15), 1)
    
    # Draw column headers
    for col in range(num_cols):
        header_x = 10 + col * col_width
        header_text = f"Column {col+1}: Addresses {col*num_rows}-{(col+1)*num_rows-1}"
        header_surf = font.render(header_text, True, (180, 180, 255))
        screen.blit(header_surf, (header_x, start_y - 25))
    
    # For each value, display in a grid layout
    for i in range(NUM_RAM_BYTES):
        col = i // num_rows
        row = i % num_rows
        
        x = 10 + col * col_width
        y = start_y + row * row_height
        
        value = values.get(i, 0)
        prev_value = last_values.get(i, value)
        change = value - prev_value
        
        # Choose text color based on changes
        if i == highlight_address:
            # Highlight the fuel address in bright green
            color = (50, 255, 50)
        elif change > 0:
            color = (50, 255, 100)  # Green for increase
        elif change < 0:
            color = (255, 100, 50)  # Red for decrease
        else:
            color = (200, 200, 200)  # Grey for no change
        
        # Format text with fixed width to prevent scrambling
        hex_addr = f"0x{i:02X}"
        dec_addr = f"{i:3d}"
        hex_val = f"{value:02X}"
        dec_val = f"{value:3d}"
        change_text = f"{change:+d}" if change != 0 else "  "
        
        # Full text with fixed width formatting
        text = f"{hex_addr}({dec_addr}): {dec_val} [{hex_val}] {change_text}"
        
        # Add a marker for the fuel address
        if i == highlight_address:
            text = "* " + text
            # Draw highlight box
            pygame.draw.rect(screen, (40, 80, 40), (x-5, y-2, col_width-10, row_height), 0)
            pygame.draw.rect(screen, (0, 255, 0), (x-5, y-2, col_width-10, row_height), 1)
        else:
            text = "  " + text
        
        # Render and display the text
        text_surf = font.render(text, True, color)
        screen.blit(text_surf, (x, y))

def display_frame(frame):
    """Convert the game frame to a pygame surface and display it"""
    global fuel_collection_events, frame_counter
    
    # Clear the screen
    screen.fill((20, 20, 30))  # Dark blue-gray background
    
    # Create a border area for the game
    pygame.draw.rect(screen, (40, 40, 60), (10, 10, 520, 480))
    pygame.draw.rect(screen, (100, 100, 150), (10, 10, 520, 480), 2)
    
    # Resize frame to fit our window
    frame = np.transpose(frame, (1, 0, 2))  # Transpose for pygame's display order
    surf = pygame.surfarray.make_surface(frame)
    surf = pygame.transform.scale(surf, (500, 450))  # Larger game view
    screen.blit(surf, (20, 25))
    
    # Get current RAM values
    current_ram_values = get_ram_values()
    
    # Add game info text
    font = pygame.font.SysFont('arial', 22)
    info_text = f"Score: {total_reward:.1f} | Press ESC to quit"
    text_surf = font.render(info_text, True, (255, 255, 255))
    screen.blit(text_surf, (550, 25))
    
    # Add frame counter
    frame_text = f"Frame: {frame_counter}"
    frame_surf = font.render(frame_text, True, (200, 200, 200))
    screen.blit(frame_surf, (550, 55))
    
    # Add fuel level indicator with a visual bar
    fuel_level = current_ram_values.get(109, 0)
    fuel_max = 150  # Approximate max fuel value
    
    # Text indicator
    fuel_text = f"Fuel Level: {fuel_level}"
    fuel_surf = font.render(fuel_text, True, (0, 255, 0))
    screen.blit(fuel_surf, (550, 85))
    
    # Visual fuel bar
    bar_width = 300
    bar_height = 25
    border_color = (150, 150, 150)
    fill_color = (0, 200, 0)
    
    # Draw border
    pygame.draw.rect(screen, border_color, (550, 115, bar_width, bar_height), 1)
    
    # Draw fill based on fuel level
    fill_width = int((fuel_level / fuel_max) * bar_width)
    pygame.draw.rect(screen, fill_color, (550, 115, fill_width, bar_height))
    
    # Track fuel level and potential fuel collection events
    for addr in CANDIDATE_FUEL_ADDRESSES:
        current_val = current_ram_values.get(addr, 0)
        fuel_history[addr].append(current_val)
        
        # Check for fuel collection events (significant increase)
        prev_val = last_ram_values.get(addr, current_val)
        if addr == 109 and current_val > prev_val + 5:  
            # Record fuel collection event
            fuel_collection_events.append((frame_counter, prev_val, current_val))
            if len(fuel_collection_events) > MAX_EVENTS:
                fuel_collection_events.pop(0)
    
    # Display RAM values in fixed layout
    draw_ram_values(screen, current_ram_values, last_ram_values)
    
    # Add controls reminder
    controls_font = pygame.font.SysFont('arial', 18)
    controls = [
        "Controls:",
        "↑ - Fire",
        "→ - Move right",
        "← - Move left", 
        "↓ - Do nothing/slow down",
        "SPACE - Alt fire",
        "ESC - Quit"
    ]
    
    for i, text in enumerate(controls):
        ctrl_surf = controls_font.render(text, True, (200, 200, 200))
        screen.blit(ctrl_surf, (800, 25 + i * 25))
    
    # Update last values for change detection
    last_ram_values.clear()  # Clear first to prevent old values persisting
    last_ram_values.update(current_ram_values)
    
    # Update display
    pygame.display.flip()

# Reset the environment
observation, info = env.reset()
frame = env.render()
total_reward = 0
done = False
action = 0  # Default action (NOOP)

# Initialize RAM values
last_ram_values = get_ram_values()

print("Controls:")
print("  ↑ (UP) - Fire")
print("  → (RIGHT) - Move right")
print("  ← (LEFT) - Move left")
print("  ↓ (DOWN) - Do nothing/slow down")
print("  SPACE - Alternative fire")
print("  ESC - Quit")
print("\nMonitoring RAM values - address 0x6D (109) is the fuel level")
print("Values that change will be highlighted")

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
    frame_counter += 1
    
    # Display the game
    display_frame(frame)
    
    # Check if episode is done
    if done or truncated:
        print(f"Episode finished! Score: {total_reward}")
        observation, info = env.reset()
        total_reward = 0
        # Reset values after restart
        last_ram_values.clear()
        last_ram_values.update(get_ram_values())
        for addr in CANDIDATE_FUEL_ADDRESSES:
            fuel_history[addr].clear()
        fuel_collection_events.clear()
        frame_counter = 0
    
    # Control game speed
    clock.tick(30)  # 30 FPS

# Clean up
env.close()
pygame.quit()