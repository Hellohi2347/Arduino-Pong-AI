import numpy as np
import os

input_data = np.zeros((5,1))
memory = []

def save_weights(file_name, point_count):
    # 1. Define the target directory
    save_dir = r"C:/Users/ldmil/Documents/Independent Project/Saves"
    
    # 2. Create the folder if it doesn't exist yet
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 3. Join the path and filename
    file_path = os.path.join(save_dir, file_name)
    
    try:
        # Save weights and the point count
        np.savez(file_path, 
                 w1=w1, b1=b1, 
                 w2=w2, b2=b2, 
                 w3=w3, b3=b3, 
                 pc=np.array([point_count]))
        # print(f"Saved to {file_path}") # Optional confirmation
    except Exception as e:
        print(f"Failed to save to {file_path}: {e}")

def load_weights(file):
    return np.load(file)

def log(data):
    return

def initialize(valw1, valb1, valw2, valb2, valw3, valb3):
    global w1,w2,w3,b1,b2,b3
    w1 = valw1
    b1 = valb1
    w2 = valw2
    b2 = valb2
    w3 = valw3
    b3 = valb3

def ReLU(z):
    return np.where(z > 0, z, z * 0.01)

def ReLU_deriv(z):
    return np.where(z > 0, 1, 0.01)

def softmax(z, T):
    return np.exp((z-np.max(z))/T)/np.sum(np.exp((z-np.max(z))/T))

def forward_pass(input, T):
    global z1,h1,z2,h2,z3,h3,input_data
    
    input_data = input.reshape(-1,1)

    z1 = np.dot(w1, input_data) + b1
    h1 = ReLU(z1)

    z2 = np.dot(w2, h1) + b2
    h2 = ReLU(z2)

    z3 = np.dot(w3, h2) + b3
    h3 = softmax(z3, T)
    return h3

def learn(rewards, alpha=0.000005, gamma=.90):
    global w1, b1, w2, b2, w3, b3, memory

    if not memory:
        return

    # 1. Discounted Rewards
    G = np.zeros_like(rewards, dtype=float)
    running_total = 0
    for t in reversed(range(len(rewards))):
        running_total = rewards[t] + gamma * running_total
        G[t] = running_total
    
    # Standardize G
    std_G = np.std(G)
    G_hat = (G - np.mean(G)) / (std_G + 1e-8)

    # 2. Vectorize Memory
    X = np.hstack([m[0] for m in memory])      # (5, N)
    Z1 = np.hstack([m[1] for m in memory])     # (H1, N)
    H1 = np.hstack([m[2] for m in memory])     # (H1, N)
    Z2 = np.hstack([m[3] for m in memory])     # (H2, N)
    H2 = np.hstack([m[4] for m in memory])     # (H2, N)
    H3 = np.hstack([m[5] for m in memory])     # (3, N)
    actions = [m[6] for m in memory]

    # Create Y_hat (One-hot targets)
    Y_hat = np.zeros_like(H3)
    Y_hat[actions, np.arange(len(actions))] = 1

    # 3. Backprop with Shape Fix
    # .reshape(1, -1) makes G_hat a (1, N) row so it broadcasts correctly
    delta3 = (H3 - Y_hat) * G_hat.reshape(1, -1) 
    
    dw3 = np.dot(delta3, H2.T)
    db3 = np.sum(delta3, axis=1, keepdims=True)

    delta2 = np.dot(w3.T, delta3) * ReLU_deriv(Z2)
    dw2 = np.dot(delta2, H1.T)
    db2 = np.sum(delta2, axis=1, keepdims=True)

    delta1 = np.dot(w2.T, delta2) * ReLU_deriv(Z1)
    dw1 = np.dot(delta1, X.T)
    db1 = np.sum(delta1, axis=1, keepdims=True)
   
    # 4. Apply Gradients
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3

    # Weight decay (L2)
    decay = 0.9999
    w1 *= decay; w2 *= decay; w3 *= decay

    memory = []