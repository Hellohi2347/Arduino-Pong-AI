import sys
import pygame
from pygame.locals import *
import time
import numpy as np
import neural_network as nn
import os
import matplotlib.pyplot as plt

# Setup the Live Plot
plt.ion() 
fig, ax = plt.subplots()
line100, = ax.plot([], [], 'r-', label="Ao100 (Short-term)")
line1000, = ax.plot([], [], 'b-', label="Ao1000 (Long-term)") # New blue line
ax.set_xlabel("Points")
ax.set_ylabel("Average Reward")
ax.legend()

# Lists to store the history so we don't have to re-calculate everything
ao100_history = []
ao1000_history = []
point_history = []

fig2, ax2 = plt.subplots()
line_hits, = ax2.plot([], [], 'g-', label="Hits Ao100")
line_temp, = ax2.plot([], [], 'y-', label="Temperature")
ax2.set_xlabel("Points")
ax2.set_ylabel("Value (Hits/Temp)")
ax2.legend()
hits_history = []
temp_history = []

# Track paddle positions (0-60)
paddle_pos_history = []

pygame.init()

FPS = 60
FramePerSec = pygame.time.Clock()
 
scale = 10
DISPLAYSURF = pygame.display.set_mode((128 * scale,64 * scale))
pygame.display.set_caption("Pong")

render_enabled = True
graphs_enabled = True  # Start with graphs on

last_time = time.time()

hits = 0
p1Y = 30
p2Y = 30
ballX = 0
ballY = 0
vX = 0
vY = 0.5
plyrCtrl = False
helpNN = False

def newV(dist):
    global vY, vX
    vY = round(4 * ((1/2 * dist)-0.875 if dist < 2 else (1/2 * dist)-1.125))/4
    if(hits == 4):
        vX = 0 - (1.5 * vX)
    elif(hits == 12):
        vX = 0 - (4/3 * vX)
    else:
        vX = 0 - vX

def reset():
    global ballX, ballY, vX, vY, hits
    ballX = 64
    ballY = 32
    if (vX != 0):
        vX = (abs(vX)/vX) * 1/2
    else: vX = -1/2
    hits = 0

def update_ball():
    global ballX, ballY, vX, vY, p1Y, p2Y, hits
    ballX += vX
    ballY += vY
    if ballX <= 1: 
        if 0 <= (ballY - p1Y) <= 4:
            ballX = 1
            hits += 1
            newV(ballY - p1Y)
            return 100.0, False
        elif ballX < -1:
            if -5 <= (ballY - p1Y) <= 9:
                return -20, True
            else:
                return -100.0, True
    elif ballX >= 127:
        if 0 <= (ballY - p2Y) <= 4:
            ballX = 126
            hits += 1
            newV(ballY - p2Y)
            return 0.0, False
        elif ballX > 130:
            return 0.0, True
    if ballY <= 0:
        ballY = 0
        vY = abs(vY)
    elif ballY >= 63:
        ballY = 63
        vY = -abs(vY) 
    return 0, False

def update_p1(action):
    global p1Y
    if (action == 2 and p1Y < 60):
        p1Y += 1
    if (action == 0 and p1Y > 0):
        p1Y -= 1

def update_p2():
    global p2Y, ballY, ballX, skill_level
    if plyrCtrl:
        if (pygame.key.get_pressed()[K_DOWN] and p2Y < 60):
            p2Y += 1
        if (pygame.key.get_pressed()[K_UP] and p2Y > 0):
            p2Y -= 1
    else:
        paddle_center = p2Y + 2
        if np.random.rand() < skill_level:
            if paddle_center-2 < ballY:
                p2Y += 1
            elif paddle_center+2 > ballY:
                p2Y -= 1
        else:
            random_choice = np.random.choice([0, 1, 2])
            if random_choice == 0:
                p2Y -= 1
            elif random_choice == 2:
                p2Y += 1

def update_heatmap():
    if not graphs_enabled: return
    window_size = 5000
    recent_history = paddle_pos_history[-window_size:]
    if len(recent_history) < 100: return
    
    plt.figure(3)
    plt.clf()
    counts, bins = np.histogram(recent_history, bins=60, range=(0, 60))
    heatmap_data = counts.reshape(-1, 1)
    
    plt.imshow(heatmap_data, cmap='magma', interpolation='gaussian', aspect=0.1, origin='lower')
    plt.title("Paddle Activity")
    # Let the log_progress handle the drawing/events

total_rewards = []
total_hits = []
point_count = 0
temperature = 0.05
min_temperature = 0.05
decay_rate = 0.999995
skill_level = 1.01
skill_level_nn = 1.00

def log_progress(reward_list):
    global point_count, total_rewards, total_hits, graphs_enabled
    # Add your history lists to the global declaration
    global point_history, ao100_history, ao1000_history, hits_history, temp_history
    
    point_count += 1
    point_reward = sum(reward_list)
    total_rewards.append(point_reward)
    total_hits.append(hits)
    paddle_pos_history.append(p1Y)

    # 1. ALWAYS update the data, regardless of if graphs are shown
    if point_count % 10 == 0:
        avg10 = sum(total_rewards[-10:]) / 10
        avg100 = sum(total_rewards[-100:]) / 100
        avg1000 = sum(total_rewards[-1000:]) / 1000
        avg_hits_100 = sum(total_hits[-100:]) / 100
        # This data must exist for the graphs to draw anything!
        point_history.append(point_count)
        ao100_history.append(avg100)
        ao1000_history.append(avg1000)
        hits_history.append(avg_hits_100)
        temp_history.append(temperature)

        print(f"Points: {point_count} | Reward Ao10: {avg10:.3f} | Reward Ao100: {avg100:.3f} | Reward Ao1000: {avg1000:.3f} | Choice: {choice} | dY: {p1Y-ballY} | T: {temperature:0.5f} | Opponent Skill: {skill_level:.2f} | Hits Ao100: {avg_hits_100:.2f}")

    # 2. Plotting Logic
    if point_count % 100 == 0 and graphs_enabled:
        # Figure 1: Rewards
        plt.figure(1)
        plt.clf()
        plt.plot(point_history, ao100_history, 'r-', label="Ao100")
        plt.plot(point_history, ao1000_history, 'b-', label="Ao1000")
        plt.legend(loc='upper left')
        plt.gcf().canvas.draw_idle() # Tells it to draw when it has a free millisecond

        # Figure 2: Hits
        plt.figure(2)
        plt.clf()
        plt.plot(point_history, hits_history, 'g-', label="Hits")
        plt.plot(point_history, temp_history, 'y-', label="Temp")
        plt.legend(loc='upper left')
        plt.gcf().canvas.draw_idle()

        update_heatmap()
        
        # Process events for all windows at once
        for i in plt.get_fignums():
            plt.figure(i).canvas.flush_events()

def load_from_file(file_name):
    file_path = os.path.join(r"C:/Users/ldmil/Documents/Independent Project/OP Saves", file_name)
    
    # Load the data object
    data = nn.load_weights(file_path)
    
    # Convert specifically to float64 to wipe out any 'string' formatting
    w1 = data["w1"].astype(np.float64)
    b1 = data["b1"].astype(np.float64)
    w2 = data["w2"].astype(np.float64)
    b2 = data["b2"].astype(np.float64)
    w3 = data["w3"].astype(np.float64)
    b3 = data["b3"].astype(np.float64)
    
    nn.initialize(w1, b1, w2, b2, w3, b3)
reset()
checkpoint_path = os.path.join(r"C:/Users/ldmil/Documents/Independent Project/OP Saves", "weights.npz")

if os.path.exists(checkpoint_path):
    checkpoint = nn.load_weights(checkpoint_path)
    nn.initialize(
        checkpoint["w1"], checkpoint["b1"], 
        checkpoint["w2"], checkpoint["b2"], 
        checkpoint["w3"], checkpoint["b3"]
    )
    # Restore point_count if you added it to the save function
    if "pc" in checkpoint:
        point_count = int(checkpoint["pc"][0])
        #point_count = 0
        # Add a starting point so the line has a place to begin
        point_history = [point_count]
        ao100_history = [0]
        ao1000_history = [0]
        hits_history = [0]
        temp_history = [temperature]
    print(f"Resuming from point {point_count}")
else:
    # Fallback to random if no save exists
    nn.initialize(np.random.randn(32,5) * np.sqrt(2/5), np.zeros((32,1)), 
                  np.random.randn(32,32) * np.sqrt(2/32), np.zeros((32,1)), 
                  np.random.randn(3,32) * np.sqrt(2/32), np.zeros((3,1)))
rewards = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            nn.save_weights("weights.npz", point_count)
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_s:
                print("Initiating emergency save...")
                nn.save_weights("weights.npz", point_count)
                print("Save complete. Closing safely.")
                time.sleep(5.0) 
                pygame.quit()
                sys.exit()
            if event.key == K_g: 
                render_enabled = not render_enabled
            if event.key == K_o: 
                helpNN = not helpNN
            if event.key == K_c:
                graphs_enabled = not graphs_enabled
                if not graphs_enabled:
                    plt.close('all') # This physically removes the windows
                else:
                    plt.ion()
            if event.key == K_p:
                plyrCtrl = not plyrCtrl
        elif event.type == KEYUP:
            if event.key == K_EQUALS and skill_level < 1:
                skill_level += 0.02
            elif event.key == K_MINUS and skill_level > 0.5:
                skill_level -= 0.02
            elif event.key == K_0 and skill_level < 1:
                skill_level_nn += 0.02
            elif event.key == K_9 and skill_level > 0.5:
                skill_level_nn -= 0.02

    # Update ball and paddles
    input = np.array([p1Y/64, ballX/128, ballY/64, vX/4, vY/4], dtype=np.float64)

    probs = nn.forward_pass(input, temperature)
    choice = np.random.choice([0,1,2], p=probs.flatten())
    actual_move = choice

    if helpNN:
        paddle_center = p1Y + 2
        if np.random.rand() < skill_level_nn:
            if paddle_center-2 < ballY:
                actual_move = 2 
            elif paddle_center+2 > ballY:
                actual_move = 0
            else:
                actual_move = 1
        else:
            actual_move = np.random.choice([0, 1, 2])
    nn.memory.append((input.reshape(-1,1), nn.z1, nn.h1, nn.z2, nn.h2, probs, actual_move))

    update_p1(actual_move)
    update_p2()

    reward, over = update_ball()

    if ballY < p1Y:
        dist = p1Y - ballY
    elif ballY > p1Y + 4:
        dist = ballY - (p1Y + 4)
    else:
        dist = 0

    #tracking_reward = .9 * np.exp(-.15 * dist)
    #reward += tracking_reward

    # New Linear Tracking: Max 0.5, Min -0.5
    tracking_reward = 0.5 - (dist / 30) 
    reward += tracking_reward

    rewards.append(reward)

    if (over):
        if temperature > min_temperature:
            temperature *= decay_rate
        log_progress(rewards)
        nn.learn(rewards)
        if (point_count % 10 == 0):
            nn.save_weights("weights.npz", point_count)
        rewards = []
        reset()
        if (point_count % 5000 == 0):
            nn.save_weights(f"large_nn_tracking_{point_count}.npz", point_count)

    current_time = time.time()
    d_time = current_time - last_time
    if (d_time != 0):
        fps = 1/d_time
        #print(f"FPS: {int(fps)}       ", end="\r")
    last_time = current_time
    
    if render_enabled:
        DISPLAYSURF.fill((0,0,0))

        # Draw middle liine
        for i in range(8):
            pygame.draw.rect(DISPLAYSURF, (255,255,255), (64 * scale, (8 * i + 2) * scale, 1 * scale, 4 * scale))

        # Draw paddles
        pygame.draw.rect(DISPLAYSURF, (255,255,255), (127 * scale, p2Y * scale, 1 * scale, 4 * scale))
        pygame.draw.rect(DISPLAYSURF, (255,0,0), (0 * scale, p1Y * scale, 1 * scale, 4 * scale))

        # Draw ball
        pygame.draw.rect(DISPLAYSURF, (255,255,255), (ballX * scale, ballY * scale, 1 * scale, 1 * scale))

        pygame.display.update()
        FramePerSec.tick(FPS)
    else:
        # This is the "Pulse" that prevents the Drag-Crash
        if graphs_enabled:
            # We process Matplotlib events every few frames
            if point_count % 5 == 0:
                for f_num in plt.get_fignums():
                    # This handles the 'dragging' messages from Windows
                    plt.figure(f_num).canvas.start_event_loop(0.000001)
            
            # Keep Pygame from thinking the app has frozen
            pygame.event.pump()