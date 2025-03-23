"# Deep Reinforcement Learning for Robotics"

## Introduction

Deep Reinforcement Learning (DRL) combines deep neural networks with reinforcement learning principles to solve complex robotic control problems. This approach enables robots to learn policies for high-dimensional state and action spaces that would be intractable with traditional RL methods.

## Foundations of Deep RL

### Neural Networks for RL
- **Function Approximation**: Replacing tabular methods with neural networks
- **Representation Learning**: Automatically extracting features from raw input
- **End-to-End Learning**: Learning directly from raw sensory input to actions
- **Generalization**: Handling continuous and high-dimensional spaces
- **Transfer Learning**: Leveraging pre-trained networks for new tasks

### Deep RL Architecture Components
- **State Representation Networks**: Convolutional, recurrent, and transformer architectures
- **Policy Networks**: Mapping states to action distributions
- **Value Networks**: Estimating state or state-action values
- **Model Networks**: Learning environment dynamics
- **Auxiliary Networks**: Predicting additional objectives for faster learning

## Core Deep RL Algorithms

### Value-Based Methods

#### Deep Q-Networks (DQN)
- **Architecture**: Neural network approximating Q-values
- **Experience Replay**: Buffer of past experiences for stable training
- **Target Networks**: Separate networks for stability
- **Variants**: Double DQN, Dueling DQN, Prioritized Experience Replay
- **Applications**: Discrete action problems in robotics

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # Target network update frequency
        self.target_update = target_update
        self.update_counter = 0
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
            
        # Sample mini-batch
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()

# Training loop example
def train_dqn(env, agent, num_episodes=1000):
    rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            
        rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")
            
    return rewards
```

#### Distributional RL
- **Value Distribution**: Learning distribution over returns
- **C51 Algorithm**: Learning categorical distribution
- **QR-DQN**: Quantile regression DQN
- **IQN**: Implicit quantile networks
- **Applications**: Handling uncertainty in robotic tasks

### Policy-Based Methods

#### Policy Gradients
- **REINFORCE**: Monte Carlo policy gradient
- **Advantage Actor-Critic (A2C/A3C)**: Reducing variance with value function
- **Applications**: Continuous control tasks

#### Trust Region Methods
- **TRPO**: Trust Region Policy Optimization
- **PPO**: Proximal Policy Optimization (clip-based)
- **Applications**: Safe robot learning, complex manipulation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(PPOActor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)
        
    def forward(self, state):
        action_mean = self.actor(state)
        return action_mean
    
    def get_distribution(self, state):
        action_mean = self.forward(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(state.device)
        return Normal(action_mean, torch.sqrt(action_var))
    
class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super(PPOCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        return self.critic(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, K_epochs=10, eps_clip=0.2, action_std_init=0.6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor-Critic networks
        self.actor = PPOActor(state_dim, action_dim, action_std_init).to(self.device)
        self.critic = PPOCritic(state_dim).to(self.device)
        
        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # Buffer
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_logprob = []
        self.buffer_reward = []
        self.buffer_done = []
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            dist = self.actor.get_distribution(state)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
            
        return action.cpu().numpy(), action_logprob.item()
    
    def store_transition(self, state, action, logprob, reward, done):
        self.buffer_state.append(state)
        self.buffer_action.append(action)
        self.buffer_logprob.append(logprob)
        self.buffer_reward.append(reward)
        self.buffer_done.append(done)
    
    def update(self):
        # Convert buffer to tensors
        old_states = torch.FloatTensor(np.array(self.buffer_state)).to(self.device)
        old_actions = torch.FloatTensor(np.array(self.buffer_action)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.buffer_logprob)).to(self.device)
        old_rewards = torch.FloatTensor(np.array(self.buffer_reward)).to(self.device)
        old_dones = torch.FloatTensor(np.array(self.buffer_done)).to(self.device)
        
        # Calculate returns
        returns = []
        discounted_return = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + (self.gamma * discounted_return)
            returns.insert(0, discounted_return)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Update policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            dist = self.actor.get_distribution(old_states)
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            state_values = self.critic(old_states).squeeze()
            
            # Find ratio (π_θ / π_θ_old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Calculate advantages
            advantages = returns - state_values.detach()
            
            # PPO update
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values, returns)
            
            # Update actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            # Update critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
        
        # Clear buffer
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_logprob = []
        self.buffer_reward = []
        self.buffer_done = []
```

### Model-Based DRL

#### Integrated Models
- **MBRL with Deep Networks**: Learning dynamics with neural networks
- **MVE**: Model-based value expansion
- **PETS**: Probabilistic ensemble trajectory sampling
- **Applications**: Sample-efficient robot learning

#### World Models
- **Predictive Models**: Learning environment dynamics
- **Latent Dynamics**: Learning compressed state representations
- **Planning in Latent Space**: Using models for planning
- **Applications**: Long-horizon planning tasks, safety-critical applications

## Advanced DRL Approaches

### Hierarchical DRL
- **Temporal Abstraction**: Learning at multiple time scales
- **Options Framework**: Learning temporally extended actions
- **Feudal Networks**: Hierarchical policy decomposition
- **Applications**: Complex long-horizon robotics tasks

### Multi-Task and Meta-Learning
- **Task Embeddings**: Learning task representations
- **Meta-RL**: Learning to learn new tasks
- **Gradient-based Meta-Learning**: MAML, Reptile
- **Applications**: Fast adaptation to new robotic tasks

### Imitation and Demonstration Learning
- **Behavioral Cloning**: Supervised learning from demonstrations
- **Inverse RL**: Inferring reward functions from demonstrations
- **GAIL**: Generative adversarial imitation learning
- **Applications**: Learning from human demonstrations

### Multi-Agent DRL
- **Cooperative Learning**: Agents working together
- **Competitive Learning**: Adversarial training
- **Mixed Scenarios**: Partial cooperation
- **Applications**: Multi-robot systems, human-robot teams

## Practical Implementations for Robotics

### Deep RL for Manipulation

#### Dexterous Manipulation
- **In-Hand Manipulation**: Learning to reorient objects
- **Tool Use**: Learning to use tools
- **Contact-Rich Manipulation**: Learning contact dynamics
- **Implementation Challenges**: Contact physics, exploration

#### Grasping
- **End-to-End Visuomotor Grasping**: From pixels to grasps
- **Shape Completion**: Learning object geometry
- **Multi-Fingered Grasping**: Learning grasp configurations
- **Implementation Approaches**: Sim-to-real, domain randomization

### Deep RL for Locomotion

#### Legged Locomotion
- **Terrain Adaptation**: Learning to navigate diverse terrains
- **Dynamic Gaits**: Learning natural gaits
- **Recovery Behaviors**: Learning to recover from falls
- **Implementation Approaches**: Curriculum learning, asymmetric training

#### Aerial Robotics
- **Agile Maneuvering**: Learning acrobatic maneuvers
- **Obstacle Avoidance**: Learning to navigate cluttered environments
- **Landing**: Learning precision landing
- **Implementation Challenges**: Sim-to-real gap, safety

### Deep RL for Navigation

#### Visual Navigation
- **End-to-End Navigation**: From camera input to control
- **Map Building**: Learning to build and use maps
- **Target-Driven Navigation**: Learning to reach goals
- **Implementation Approaches**: Auxiliary tasks, representation learning

#### Social Navigation
- **Human-Aware Navigation**: Learning socially acceptable behavior
- **Crowd Navigation**: Learning to navigate crowded spaces
- **Communication**: Learning to signal intentions
- **Implementation Challenges**: Human modeling, safety

## Implementation Considerations

### Sim-to-Real Transfer

#### Domain Randomization
- **State Randomization**: Varying physical parameters
- **Visual Randomization**: Varying visual appearances
- **Dynamics Randomization**: Varying dynamic properties
- **Implementation Tools**: PyBullet, MuJoCo, Isaac Gym

#### System Identification
- **Model Adaptation**: Adapting simulation to reality
- **Online Fine-Tuning**: Adapting policies in the real world
- **Implementation Approaches**: Bayesian optimization, meta-learning

### Exploration in Deep RL

#### Intrinsic Motivation
- **Curiosity-Driven Exploration**: Rewarding novelty
- **Prediction Error**: Rewarding learning progress
- **Count-Based Exploration**: Rewarding visiting new states
- **Implementation Examples**: Random Network Distillation (RND), ICM

#### Curriculum Learning
- **Task Progression**: Gradually increasing difficulty
- **Automatic Curricula**: Self-generated tasks
- **Teacher-Student Framework**: Learning what to learn
- **Implementation Approaches**: Reverse curriculum, asymptotic learning

### Safety in Deep RL

#### Safe Exploration
- **Constrained RL**: Learning with constraints
- **Risk-Sensitive RL**: Accounting for worst-case scenarios
- **Uncertainty Estimation**: Knowing what the agent doesn't know
- **Implementation Approaches**: Lagrangian methods, shielding

#### Sim-to-Real Safety
- **Conservative Policies**: Ensuring safety in transfer
- **Reality Gap Robustness**: Handling model mismatch
- **Implementation Tools**: Safety layers, teacher oversight

## Deep RL Frameworks and Libraries

### Specialized Robotics Frameworks
- **OpenAI Gym Robotics**: Standard environments
- **PyBullet**: Physics simulation and RL
- **TF-Agents/Stable Baselines**: High-level RL implementations
- **Isaac Gym**: GPU-accelerated robot learning
- **Getting Started**: Setting up training pipelines

### Evaluation and Benchmarking
- **Standard Benchmarks**: D4RL, Meta-World, RLBench
- **Metrics**: Success rate, sample efficiency, robustness
- **Reporting Practices**: Reproducibility guidelines
- **Implementation Approach**: Standardized evaluation protocols

## Project Examples

### Example Project 1: DQN for Object Sorting
Implementation of a DQN agent for sorting objects using a robotic arm.

### Example Project 2: PPO for Legged Locomotion
Using PPO to learn stable walking gaits for a quadruped robot.

### Example Project 3: Model-Based RL for Dexterous Manipulation
Implementing a model-based approach for in-hand object manipulation.

### Example Project 4: RL with Human Feedback
Incorporating human preferences into the learning process for a service robot.

## Future Directions

### Foundation Models for Robotics
- **Generalist Robots**: Learning general-purpose policies
- **Large-Scale Pretraining**: Learning from diverse data
- **Multimodal Representations**: Combining vision, language, and action
- **Research Directions**: RT-X, RT-1, PaLM-E

### Embodied Intelligence
- **Grounded Language Learning**: Connecting language to actions
- **Affordance Learning**: Discovering action possibilities
- **Common Sense Reasoning**: Understanding physical world constraints
- **Research Examples**: SayCan, PaLM-E

### Lifelong Learning Robots
- **Continual Learning**: Avoiding catastrophic forgetting
- **Knowledge Accumulation**: Building on previous experiences
- **Skill Libraries**: Reusing learned behaviors
- **Open Challenges**: Knowledge transfer, forgetting prevention

## Resources and Further Learning

### Key Papers
- Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
- Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
- Andrychowicz et al. "Learning Dexterous In-Hand Manipulation" (2018)
- Levine et al. "Learning Hand-Eye Coordination for Robotic Grasping" (2016)

### Online Courses
- DeepMind's RL Course
- Berkeley's Deep RL Course (CS285)
- Stanford's Reinforcement Learning for Robotics

### Recommended Books
- Sutton & Barto: Reinforcement Learning: An Introduction
- Kober, Bagnell & Peters: Reinforcement Learning in Robotics: A Survey

## Related Topics
- [Basic RL](../6.1_Basic_RL/README.md): Foundational reinforcement learning concepts
- [Computer Vision](../../03_Perception/README.md): Visual perception for RL
- [Motion Planning](../../05_Motion_Planning/README.md): Classical planning approaches
- [Robot Learning](../../08_Machine_Learning_for_Robotics/README.md): Other learning approaches