"""Quick training test - 50 episodes"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from factory_layout_rl import (
    FactoryLayoutEnvironment,
    DDQLAgent,
    train_layout_optimizer,
    plot_training_results
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("="*70)
print("Quick Training Test - 50 Episodes")
print("="*70)

# Train for just 50 episodes
agent, env, episode_rewards, moving_avg_rewards, best_layout, best_reward = train_layout_optimizer(episodes=50)

print("\n" + "="*70)
print(f"Training Complete! Best Reward: {best_reward:.2f}")
print("="*70)

# Save plots
print("\nSaving results...")
fig1 = plot_training_results(episode_rewards, moving_avg_rewards)
plt.savefig('quick_training.png', dpi=150, bbox_inches='tight')
print("Saved training progress to 'quick_training.png'")

env.grid = best_layout
fig2 = env.render()
plt.savefig('quick_layout.png', dpi=150, bbox_inches='tight')
print("Saved best layout to 'quick_layout.png'")

plt.close('all')

print("\n" + "="*70)
print("Quick test complete! Files saved.")
print("="*70)
