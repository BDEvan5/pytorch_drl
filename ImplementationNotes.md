# Implementation Notes

## Algorithms

### Soft Actor Critic (SAC):

There are several variants of the SAC algorithm. 
The one in this repo uses a stochaistic policy and automatic tuning of the alpha parameter.
It also uses two Q-networks.

The SAC uses a different policy network compared to the DDPG and TD3 algorithms because of the reparameterisation trick that is applied.
The SAC policy has two heads that calculate a mean and a standard deviation repsectively.
The action is then sampled according to these parameters.




