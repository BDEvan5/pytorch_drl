# BaselineDRL Implementations

The aim of this repository is to provide implementations common DRL algorithms that can be easily transferred to other engineering applications.

The document `BaselineDRL.pdf` provides an introduction to deep reinforcement learning and the important formulas behind the algorithms.

The repository and document are still under construction and will be update periodically in the upcoming weeks.

## Structure

The `IndependantAlgs` folder contains files with working examples of each algorithm.

Currently the following algorithms are completed:
- DQN: Deep-Q-network
- PG: Policy gradient algorithm
- A2C: Advantage actor critic
- A2C_ent: Advantage actor critic with entropy loss
- PPO: Proximal policy optimisation

The following algorithms are under construction:
- DDPG: Deep deterministic policy gradient
- TD3: Twin-delayed-DDPG
- SAC: Soft actor critic

## Planned Projects

### Component Standardisation

It is planned to split the algorithms up into standard components for easy comparision.
The split will take place in the categories of Components and Components algorithms since there is a significant deviation between the two.
The following components will be used:
- Replay replay_buffer
- Training loop
- Network
- Learning algorithm

### Benchmark Experiments

The second improvment is to generate example experiments using the standard formats.
This is done to:
1. Benchmark these implementations
2. Enable easy tuning and experimentstion
3. Promote good scientific experiementation in RL algorithms, e.g. random seeding
4. Provide a template for experiments in other domains.


## Ideas

Things that could be implemented subject to time...

- Remove all hyperparameters from files and create a central yaml file with all hyper parameters

