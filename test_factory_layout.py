"""Quick test of factory layout RL implementation"""
import sys
print("Starting imports...", flush=True)

import tensorflow as tf
print("TensorFlow imported", flush=True)

import numpy as np
print("NumPy imported", flush=True)

# Import the main classes
from factory_layout_rl import FactoryLayoutEnvironment, DDQLAgent
print("Classes imported successfully!", flush=True)

# Test environment
print("\n" + "="*50)
print("Testing Environment...")
print("="*50)

env = FactoryLayoutEnvironment()
print(f"Environment created: {env.grid_width}x{env.grid_height} grid")

state = env.reset()
print(f"State shape: {state['grid'].shape}")
print(f"Action space size: {env.get_action_space_size()}")

# Test a few random actions
print("\nTesting random actions...")
for i in range(5):
    action = np.random.randint(0, env.get_action_space_size())
    next_state, reward, done = env.step(action)
    print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={done}")
    if done:
        break

print("\n" + "="*50)
print("Environment test completed successfully!")
print("="*50)

# Test agent creation
print("\nTesting Agent Creation...")
state_shape = (env.grid_height * env.grid_width + 4,)
action_size = env.get_action_space_size()

print(f"Creating agent with state_shape={state_shape}, action_size={action_size}")
agent = DDQLAgent(state_shape, action_size)
print("Agent created successfully!")

# Test agent action selection
state = env.reset()
action = agent.get_action(state, episode=0)
print(f"Agent selected action: {action}")

print("\n" + "="*50)
print("ALL TESTS PASSED!")
print("="*50)
print("\nThe full implementation is ready to train.")
print("Run 'python factory_layout_rl.py' to start training.")
