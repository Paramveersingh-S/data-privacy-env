# 🏋️ Training Guide

To train an agent on `DataPrivacyEnv`, you can use the standard `openenv-core` interface.

## 🛠️ Basic Training Loop (Pseudocode)
```python
from openenv_core import make
from stable_baselines3 import PPO

# Initialize the environment
env = make("data-privacy-env")

# Initialize a RL model (e.g., Proximal Policy Optimization)
model = PPO("MultiInputPolicy", env, verbose=1)

# Train for 10,000 steps
model.learn(total_timesteps=10000)

# Save the trained agent
model.save("pii_redactor_agent")
📈 Benchmarking
We recommend training across all three tasks (easy, medium, hard) to ensure the agent learns generalizable redaction strategies rather than just hardcoding specific log IDs.
