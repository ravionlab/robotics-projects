# Basic Reinforcement Learning

## Introduction
Basic Reinforcement Learning (RL) provides the foundation for developing intelligent robotics systems that can learn from interaction with their environment. This section covers the fundamental concepts, algorithms, and applications of RL for robotics problems.

## Markov Decision Processes

### MDP Framework
- **States (S)**: Robot and environment configuration
- **Actions (A)**: Control inputs available to the robot
- **Transition Function P(s'|s,a)**: Probability of reaching state s' from state s by action a
- **Reward Function R(s,a,s')**: Feedback signal after transition
- **Discount Factor γ**: Weight of future rewards (0 ≤ γ ≤ 1)
- **Horizon**: Finite or infinite time steps

### Formulating Robotic Tasks as MDPs
- **State Space Design**: Choosing relevant state variables
- **Action Space Design**: Defining available control actions
- **Reward Shaping**: Crafting rewards to encourage desired behavior
- **Discount Factor Selection**: Balancing immediate vs. future rewards
- **Example**: Mobile robot navigation, arm positioning, grasping

```python
def simple_navigation_mdp():
    """Example of simple navigation MDP definition"""
    # State space: 2D grid positions (x, y)
    states = [(x, y) for x in range(10) for y in range(10)]
    
    # Action space: Move in four directions
    actions = ['up', 'down', 'left', 'right']
    
    # Goal state
    goal = (9, 9)
    
    # Obstacles
    obstacles = [(2, 2), (2, 3), (3, 2), (7, 7)]
    
    # Reward function
    def reward(state, action, next_state):
        if next_state == goal:
            return 100  # Reaching goal
        if next_state in obstacles:
            return -100  # Hitting obstacle
        return -1  # Step cost
    
    # Transition function (deterministic in this case)
    def transition(state, action):
        x, y = state
        if action == 'up':
            next_state = (x, min(y+1, 9))
        elif action == 'down':
            next_state = (x, max(y-1, 0))
        elif action == 'left':
            next_state = (max(x-1, 0), y)
        elif action == 'right':
            next_state = (min(x+1, 9), y)
        
        # Check if next state is an obstacle
        if next_state in obstacles:
            return state  # Stay in place
        return next_state
    
    return states, actions, transition, reward, goal
```

## Value Functions and Policies

### Value Functions
- **State Value Function V(s)**: Expected return from state s
- **Action Value Function Q(s,a)**: Expected return from taking action a in state s
- **Bellman Equations**: Recursive relationships for value functions
  - V(s) = max_a [R(s,a) + γ ∑_s' P(s'|s,a)V(s')]
  - Q(s,a) = R(s,a) + γ ∑_s' P(s'|s,a)max_a' Q(s',a')

### Policies
- **Deterministic Policy π(s)**: Mapping from states to actions
- **Stochastic Policy π(a|s)**: Probability distribution over actions
- **Optimal Policy π***: Policy that maximizes expected return
- **Exploration vs. Exploitation**:
  - ε-greedy: Choose random action with probability ε
  - Softmax: Probabilistic action selection based on value estimates
  - UCB: Upper Confidence Bound for exploration

## Foundational RL Algorithms

### Dynamic Programming
- **Policy Evaluation**: Computing value function for a policy
- **Policy Improvement**: Improving policy based on value function
- **Policy Iteration**: Alternating between evaluation and improvement
- **Value Iteration**: Directly computing optimal value function
- **Applications**: Problems with known dynamics (simulation)

```python
def value_iteration(states, actions, transition, reward, gamma=0.9, theta=0.01):
    """
    Value Iteration algorithm for finding optimal policy.
    
    Args:
        states: List of states
        actions: List of actions
        transition: Function mapping (state, action) to next_state
        reward: Function mapping (state, action, next_state) to reward
        gamma: Discount factor
        theta: Convergence threshold
        
    Returns:
        V: Optimal value function
        pi: Optimal policy
    """
    # Initialize value function
    V = {s: 0 for s in states}
    
    while True:
        delta = 0
        
        # Update value function for each state
        for s in states:
            v = V[s]
            
            # Find maximum value across all actions
            action_values = []
            for a in actions:
                s_prime = transition(s, a)
                r = reward(s, a, s_prime)
                action_values.append(r + gamma * V[s_prime])
            
            V[s] = max(action_values) if action_values else 0
            delta = max(delta, abs(v - V[s]))
        
        # Check convergence
        if delta < theta:
            break
    
    # Extract policy
    pi = {}
    for s in states:
        best_action = None
        best_value = float('-inf')
        
        for a in actions:
            s_prime = transition(s, a)
            r = reward(s, a, s_prime)
            value = r + gamma * V[s_prime]
            
            if value > best_value:
                best_value = value
                best_action = a
        
        pi[s] = best_action
    
    return V, pi
```

### Monte Carlo Methods
- **Episode-Based Learning**: Learning from complete episodes
- **First-Visit MC**: Update based on first occurrence of state in episode
- **Every-Visit MC**: Update based on every occurrence
- **Exploring Starts**: Ensuring exploration of all state-action pairs
- **Applications**: Episodic tasks, when model is unknown

### Temporal Difference Learning
- **TD(0)**: One-step bootstrapping
  - V(s) ← V(s) + α[r + γV(s') - V(s)]
- **SARSA**: On-policy TD control
  - Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- **Q-Learning**: Off-policy TD control
  - Q(s,a) ← Q(s,a) + α[r + γmax_a' Q(s',a') - Q(s,a)]
- **Applications**: Control problems, continuous learning

```python
def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-Learning algorithm for robot control.
    
    Args:
        env: Environment with step(action) -> next_state, reward, done
        num_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        
    Returns:
        Q: Learned Q-function
    """
    import numpy as np
    
    # Initialize Q-table
    Q = {}
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        done = False
        
        while not done:
            # Ensure state is hashable
            state_key = tuple(state) if isinstance(state, np.ndarray) else state
            
            # Initialize state if not seen before
            if state_key not in Q:
                Q[state_key] = {a: 0 for a in range(env.action_space.n)}
            
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = max(Q[state_key], key=Q[state_key].get)  # Greedy action
            
            # Take action and observe result
            next_state, reward, done, _ = env.step(action)
            next_state_key = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
            
            # Initialize next state if not seen before
            if next_state_key not in Q:
                Q[next_state_key] = {a: 0 for a in range(env.action_space.n)}
            
            # Q-learning update
            best_next_action = max(Q[next_state_key], key=Q[next_state_key].get)
            td_target = reward + gamma * Q[next_state_key][best_next_action]
            td_error = td_target - Q[state_key][action]
            Q[state_key][action] += alpha * td_error
            
            # Move to next state
            state = next_state
    
    return Q
```

## Function Approximation

### Linear Function Approximation
- **Feature Vectors**: Representing states with feature vectors
- **Linear Value Function**: V(s) = w·φ(s) or Q(s,a) = w·φ(s,a)
- **Linear Regression**: Finding weights to minimize error
- **Gradient TD Methods**: Stability in off-policy learning
- **Suitable Features**: Polynomial, Fourier basis, tile coding

### Non-Linear Function Approximation
- **Decision Trees**: Partitioning state space
- **Neural Networks**: Multilayer perceptrons for value approximation
- **Kernel Methods**: Radial basis functions, Gaussian processes
- **Challenges**: Convergence issues, instability
- **Bridge to Deep RL**: Foundation for neural network approaches

## Applications to Basic Robotic Tasks

### Simple Navigation
- **Grid World**: 2D navigation with discrete states
- **Continuous Navigation**: Path planning in continuous spaces
- **Obstacle Avoidance**: Learning to navigate around obstacles
- **Reward Design**: Sparse vs. dense rewards for reaching goals
- **State Representation**: Position, velocity, sensor readings

### Basic Manipulation
- **Reaching**: Moving end-effector to target position
- **Pushing**: Learning to push objects to targets
- **Simple Grasping**: Binary gripper control
- **State Space**: Joint angles, end-effector position, object position
- **Action Space**: Joint velocities or torques

### Discrete Control Problems
- **Robot Decision Making**: High-level task selection
- **Discrete State/Action Spaces**: Simplified problem representations
- **Hierarchical Control**: Combining RL with classical controllers
- **Behavior Trees**: Integrating learned behaviors in larger systems
- **Human-Robot Interaction**: Learning interaction policies

## Practical Implementation

### RL Workflow
1. **Problem Formulation**: Define states, actions, rewards
2. **Algorithm Selection**: Choose appropriate method
3. **Environment Setup**: Simulation or real-world setup
4. **Training**: Run algorithm, tune hyperparameters
5. **Evaluation**: Test learned policy
6. **Deployment**: Transfer to real system (if trained in simulation)

### Environment Design
- **Simulation Tools**: OpenAI Gym, PyBullet, MuJoCo
- **Custom Environments**: Creating task-specific environments
- **Reward Shaping**: Designing rewards to guide learning
- **Reset Conditions**: Defining episode termination criteria
- **Observation Design**: Selecting relevant state information

### RL Project Structure
```
project/
├── environment/
│   ├── __init__.py
│   ├── robot_env.py     # Custom environment
│   └── rewards.py       # Reward functions
├── agents/
│   ├── __init__.py
│   ├── q_learning.py    # Q-learning implementation
│   └── sarsa.py         # SARSA implementation
├── utils/
│   ├── __init__.py
│   ├── visualization.py # Plotting utilities
│   └── evaluation.py    # Policy evaluation tools
└── main.py              # Training and evaluation script
```

## Learning Challenges and Solutions

### Sample Efficiency
- **Experience Replay**: Reusing past experiences
- **Prioritized Sweeping**: Focusing updates on significant states
- **Model-Based RL**: Learning environment dynamics
- **Transfer Learning**: Starting from pre-trained policies
- **Imitation Learning**: Learning from demonstrations

### Exploration Strategies
- **ε-Greedy with Decay**: Reducing exploration over time
- **Optimistic Initialization**: Encouraging exploration of unvisited states
- **Count-Based Exploration**: Rewarding visits to less-seen states
- **Parameter Noise**: Adding noise to policy parameters
- **Intrinsic Motivation**: Curiosity-driven exploration

### Practical Considerations
- **Hyperparameter Tuning**: Learning rate, discount factor, exploration rate
- **Feature Engineering**: Designing good state representations
- **Reward Design**: Creating informative reward signals
- **Convergence Issues**: Detecting and addressing learning problems
- **Evaluation Metrics**: Assessing policy performance

## Exercises and Projects

### Exercise 1: Grid World Navigation
Implement Q-learning for a robot navigating a 2D grid with obstacles.

### Exercise 2: Continuous Control
Apply SARSA with function approximation to a continuous state space problem.

### Exercise 3: Reward Shaping
Experiment with different reward functions for a robotic reaching task.

### Exercise 4: Policy Visualization
Create visualizations of learned policies and value functions.

## Related Topics
- [Deep RL](../6.2_Deep_RL/README.md): Neural network based approaches
- [Control Systems](../../02_Control_Systems/README.md): Classical control methods
- [Perception](../../03_Perception/README.md): State estimation for RL
- [Simulation](../../07_Robotics_Software_Frameworks/README.md): Training environments 
