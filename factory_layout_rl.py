"""
Factory Layout Optimization using Reinforcement Learning
Based on the paper: "An implementation of a reinforcement learning based algorithm
for factory layout planning" by Klar, Glatt, and Aurich (2021)

This implementation uses Double Deep Q-Learning (DDQL) to optimize the placement
of 4 functional units in a factory layout to minimize transportation time.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# Constants for grid states
FREE = 0
WALL = 1
LANE = 2
UNIT_1 = 3
UNIT_2 = 4
UNIT_3 = 5
UNIT_4 = 6


class FunctionalUnit:
    """Represents a functional unit (machine) to be placed in the layout"""
    def __init__(self, unit_id, width, height):
        self.unit_id = unit_id
        self.width = width
        self.height = height

    def get_rotated_dimensions(self, rotation):
        """Get dimensions after rotation (0 or 90 degrees)"""
        if rotation == 0:
            return self.width, self.height
        else:  # 90 degrees
            return self.height, self.width


class FactoryLayoutEnvironment:
    """Environment for factory layout planning"""

    def __init__(self, grid_width=10, grid_height=8):
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Define the 4 functional units from the case study
        self.functional_units = [
            FunctionalUnit(1, 3, 1),  # Unit 1: 3x1
            FunctionalUnit(2, 2, 1),  # Unit 2: 2x1
            FunctionalUnit(3, 2, 2),  # Unit 3: 2x2
            FunctionalUnit(4, 1, 1),  # Unit 4: 1x1
        ]

        # Transportation intensity (AGV trips) from the paper
        # Format: [incoming_goods, unit1, unit2, unit3, unit4, outgoing_goods]
        self.transport_intensity = np.array([
            [0, 10, 0, 0, 0, 0],      # From incoming goods
            [0, 0, 10, 0, 0, 0],      # From unit 1
            [0, 0, 0, 30, 0, 0],      # From unit 2
            [0, 0, 0, 0, 20, 0],      # From unit 3
            [0, 0, 0, 0, 0, 10],      # From unit 4
            [0, 0, 0, 0, 0, 0]        # From outgoing goods
        ])

        self.reset()

    def reset(self):
        """Reset the environment to initial state"""
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)

        # Set walls (borders)
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        # Set lane (middle horizontal lane for AGV)
        lane_row = self.grid_height // 2
        self.grid[lane_row, 1:-1] = LANE

        self.current_unit_idx = 0
        self.placed_units = []  # List of (unit_id, x, y, width, height)
        self.done = False

        return self._get_state()

    def _get_state(self):
        """Get current state representation"""
        # State includes: grid representation + next unit info
        state = {
            'grid': self.grid.copy(),
            'next_unit_idx': self.current_unit_idx,
            'next_unit': self.functional_units[self.current_unit_idx] if self.current_unit_idx < len(self.functional_units) else None
        }
        return state

    def _is_valid_placement(self, x, y, width, height):
        """Check if placement is valid"""
        # Check bounds
        if x < 0 or y < 0 or x + width > self.grid_width or y + height > self.grid_height:
            return False

        # Check if space is free
        for i in range(y, y + height):
            for j in range(x, x + width):
                if self.grid[i, j] != FREE:
                    return False

        # Check if at least one side is adjacent to the lane
        adjacent_to_lane = False
        for i in range(y, y + height):
            for j in range(x, x + width):
                # Check all 4 neighbors
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < self.grid_height and 0 <= nj < self.grid_width:
                        if self.grid[ni, nj] == LANE:
                            adjacent_to_lane = True
                            break
                if adjacent_to_lane:
                    break
            if adjacent_to_lane:
                break

        return adjacent_to_lane

    def _place_unit(self, x, y, width, height, unit_id):
        """Place a unit on the grid"""
        for i in range(y, y + height):
            for j in range(x, x + width):
                self.grid[i, j] = unit_id + 2  # Offset to avoid conflict with FREE, WALL, LANE

        self.placed_units.append({
            'unit_id': unit_id,
            'x': x,
            'y': y,
            'width': width,
            'height': height
        })

    def _calculate_transportation_time(self):
        """Calculate total transportation time based on distances and intensities"""
        if len(self.placed_units) != len(self.functional_units):
            return float('inf')

        # Calculate center positions for each unit
        centers = {}
        for unit in self.placed_units:
            center_x = unit['x'] + unit['width'] / 2
            center_y = unit['y'] + unit['height'] / 2
            centers[unit['unit_id']] = (center_x, center_y)

        # Add incoming/outgoing goods positions (at lane entrances)
        lane_row = self.grid_height // 2
        centers['incoming'] = (1, lane_row)  # Incoming goods (left side, just inside wall)
        centers['outgoing'] = (self.grid_width - 2, lane_row)  # Outgoing goods (right side, just inside wall)

        total_time = 0

        # Calculate transportation time between all pairs
        # Using Manhattan distance as approximation for AGV travel
        # Transport intensity matrix format: [incoming, unit1, unit2, unit3, unit4, outgoing]
        location_map = ['incoming', 1, 2, 3, 4, 'outgoing']

        for i in range(6):
            for j in range(6):
                if self.transport_intensity[i, j] > 0:
                    from_loc = location_map[i]
                    to_loc = location_map[j]

                    from_center = centers[from_loc]
                    to_center = centers[to_loc]

                    distance = abs(from_center[0] - to_center[0]) + abs(from_center[1] - to_center[1])
                    total_time += distance * self.transport_intensity[i, j]

        return total_time

    def step(self, action):
        """Execute action and return next state, reward, done"""
        if self.done:
            return self._get_state(), 0, True

        # Decode action: action = position * 2 + rotation
        position = action // 2
        rotation = action % 2  # 0 or 90 degrees

        # Convert position to x, y coordinates
        x = position % self.grid_width
        y = position // self.grid_width

        current_unit = self.functional_units[self.current_unit_idx]
        width, height = current_unit.get_rotated_dimensions(rotation * 90)

        # Check if placement is valid
        if self._is_valid_placement(x, y, width, height):
            # Place the unit
            self._place_unit(x, y, width, height, current_unit.unit_id)

            # Move to next unit
            self.current_unit_idx += 1

            # Check if all units are placed
            if self.current_unit_idx >= len(self.functional_units):
                self.done = True
                # Calculate final reward based on transportation time
                transport_time = self._calculate_transportation_time()

                # Conservative estimation for min/max (from experimentation)
                # These values should be tuned based on problem specifics
                transport_time_min = 200
                transport_time_max = 600

                normalized_time = (transport_time - transport_time_min) / (transport_time_max - transport_time_min)
                reward = 1 - np.clip(normalized_time, 0, 1)
            else:
                reward = 0  # Intermediate reward
        else:
            # Invalid placement
            reward = -1
            # Don't advance to next unit

        return self._get_state(), reward, self.done

    def get_action_space_size(self):
        """Get total number of possible actions"""
        return self.grid_width * self.grid_height * 2  # 2 rotations per position

    def render(self):
        """Visualize the current layout"""
        plt.figure(figsize=(12, 8))

        # Create colored grid
        display_grid = self.grid.copy().astype(float)

        # Color mapping
        cmap = plt.cm.get_cmap('tab10')

        plt.imshow(display_grid, cmap='tab10', vmin=0, vmax=10)
        plt.colorbar(label='Cell Type')
        plt.title('Factory Layout')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)

        # Add labels for placed units
        for unit in self.placed_units:
            center_x = unit['x'] + unit['width'] / 2
            center_y = unit['y'] + unit['height'] / 2
            plt.text(center_x, center_y, f"U{unit['unit_id']}",
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        plt.tight_layout()
        return plt.gcf()


class ReplayBuffer:
    """Experience replay buffer for DDQL"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)

    def size(self):
        """Get current buffer size"""
        return len(self.buffer)


class DDQLAgent:
    """Double Deep Q-Learning Agent"""

    def __init__(self, state_shape, action_size, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.4  # Initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay_episodes = 8000

        # Create main and target networks
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_network(self):
        """Build neural network for Q-value approximation"""
        model = models.Sequential([
            layers.Input(shape=self.state_shape),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        return model

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.main_network.get_weights())

    def get_action(self, state, episode):
        """Select action using epsilon-greedy policy"""
        # Decay epsilon
        epsilon = max(self.epsilon_min,
                     self.epsilon - (self.epsilon - self.epsilon_min) * episode / self.epsilon_decay_episodes)

        if np.random.rand() < epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_size)
        else:
            # Exploit: best action from Q-network
            state_tensor = self._preprocess_state(state)
            q_values = self.main_network(state_tensor, training=False)
            return np.argmax(q_values[0])

    def _preprocess_state(self, state):
        """Preprocess state for network input"""
        grid = state['grid']
        next_unit_idx = state['next_unit_idx']

        # One-hot encode the unit index
        unit_encoding = np.zeros(4)
        if next_unit_idx < 4:
            unit_encoding[next_unit_idx] = 1

        # Flatten grid and concatenate with unit encoding
        flattened = np.concatenate([grid.flatten(), unit_encoding])
        return tf.expand_dims(flattened, 0)

    def train(self):
        """Train the network on a batch from replay buffer"""
        if self.replay_buffer.size() < self.batch_size:
            return 0

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, done in batch:
            state_tensor = self._preprocess_state(state)
            next_state_tensor = self._preprocess_state(next_state)

            # Get current Q-values
            q_values = self.main_network(state_tensor, training=False).numpy()

            if done:
                target = reward
            else:
                # Double DQN: use main network to select action, target network to evaluate
                next_q_main = self.main_network(next_state_tensor, training=False).numpy()
                best_action = np.argmax(next_q_main[0])

                next_q_target = self.target_network(next_state_tensor, training=False).numpy()
                target = reward + self.gamma * next_q_target[0][best_action]

            q_values[0][action] = target

            states.append(state_tensor[0].numpy())
            targets.append(q_values[0])

        states = np.array(states)
        targets = np.array(targets)

        # Train the network
        with tf.GradientTape() as tape:
            predictions = self.main_network(states, training=True)
            loss = tf.keras.losses.MSE(targets, predictions)

        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))

        return float(loss.numpy().mean())

    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)


def train_layout_optimizer(episodes=8000):
    """Train the factory layout optimizer"""
    env = FactoryLayoutEnvironment()
    state_shape = (env.grid_height * env.grid_width + 4,)  # Grid + unit encoding
    action_size = env.get_action_space_size()

    agent = DDQLAgent(state_shape, action_size)

    # Training metrics
    episode_rewards = []
    moving_avg_rewards = []
    losses = []
    best_reward = -float('inf')
    best_layout = None

    print(f"Starting training for {episodes} episodes...")
    print(f"State shape: {state_shape}, Action space: {action_size}\n")

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.get_action(state, episode)

            # Take action
            next_state, reward, done = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train the agent
            loss = agent.train()
            if loss > 0:
                losses.append(loss)

            episode_reward += reward
            state = next_state

        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        # Track metrics
        episode_rewards.append(episode_reward)

        # Calculate moving average
        window = min(100, episode + 1)
        moving_avg = np.mean(episode_rewards[-window:])
        moving_avg_rewards.append(moving_avg)

        # Save best layout
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_layout = env.grid.copy()

        # Print progress
        if (episode + 1) % 10 == 0 or episode < 5:  # Print first 5, then every 10
            epsilon = max(agent.epsilon_min,
                         agent.epsilon - (agent.epsilon - agent.epsilon_min) * episode / agent.epsilon_decay_episodes)
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward: {moving_avg:.2f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Best: {best_reward:.2f}", flush=True)

    return agent, env, episode_rewards, moving_avg_rewards, best_layout, best_reward


def plot_training_results(episode_rewards, moving_avg_rewards):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot episode rewards
    axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    axes[0].plot(moving_avg_rewards, linewidth=2, label='Moving Average (100 episodes)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Progress: Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot moving average only (clearer view)
    axes[1].plot(moving_avg_rewards, linewidth=2, color='orange')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Moving Average Reward')
    axes[1].set_title('Training Progress: Moving Average Reward (100 episodes)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Configuration
    QUICK_TEST = True  # Set to False for full training (8000 episodes)
    EPISODES = 200 if QUICK_TEST else 8000

    # Train the agent
    print("=" * 70)
    print("Factory Layout Optimization using Double Deep Q-Learning")
    print("Based on: Klar et al. (2021) - Manufacturing Letters")
    print("=" * 70)
    if QUICK_TEST:
        print("QUICK TEST MODE - Running 200 episodes for faster testing")
        print("Set QUICK_TEST = False for full 8000 episode training")
    print("=" * 70 + "\n")

    agent, env, episode_rewards, moving_avg_rewards, best_layout, best_reward = train_layout_optimizer(episodes=EPISODES)

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Reward: {best_reward:.2f}")
    print("=" * 70)

    # Plot training results
    print("\nGenerating training plots...")
    fig1 = plot_training_results(episode_rewards, moving_avg_rewards)
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Saved training progress to 'training_progress.png'")

    # Visualize best layout
    print("\nGenerating best layout visualization...")
    env.grid = best_layout
    fig2 = env.render()
    plt.savefig('best_layout.png', dpi=150, bbox_inches='tight')
    print("Saved best layout to 'best_layout.png'")

    # Test the trained agent
    print("\nTesting trained agent...")
    state = env.reset()
    done = False
    test_reward = 0

    while not done:
        action = agent.get_action(state, episode=10000)  # Use greedy policy
        state, reward, done = env.step(action)
        test_reward += reward

    print(f"Test episode reward: {test_reward:.2f}")

    # Visualize test layout
    fig3 = env.render()
    plt.savefig('test_layout.png', dpi=150, bbox_inches='tight')
    print("Saved test layout to 'test_layout.png'")

    plt.close('all')  # Close all figures to free memory

    print("\n" + "=" * 70)
    print("All done! Check the generated PNG files for results.")
    print("=" * 70)
