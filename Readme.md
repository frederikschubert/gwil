# Re-Implementation of Cross-Domain Imitation Learning via Optimal Transport

The original paper is currently under review at [ICLR 2022](https://openreview.net/forum?id=xP3cPq2hQC).

## Core Idea

Minimising the Gromov-Wasserstein distance between the trajectory of an expert in one domain and an agent in a (possibly) different domain allows the agent to recover
the expert's policy up to an isometry (rotation, translation, reflection). 

## Progress

- Added the GW calculation to a DQN agent on `CartPole-v0` and try to immitate a static expert policy.
    - First experiments reach an average return of 50 after 7500 steps. 
