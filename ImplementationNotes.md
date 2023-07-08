# Implementation Notes

## Algorithms

### Soft Actor Critic (SAC):

There are several variants of the SAC algorithm. 
The one in this repo uses a stochaistic policy and automatic tuning of the alpha parameter.
It also uses two Q-networks.

The SAC uses a different policy network compared to the DDPG and TD3 algorithms because of the reparameterisation trick that is applied.
The SAC policy has two heads that calculate a mean and a standard deviation repsectively.
The action is then sampled according to these parameters.


## PyTorch 

### Gradient Updates

To update the parameters of a neural network, you must calculate a loss and then use the optimiser to update the network.
An example is provided.
```Python
    loss_fcn = torch.nn.MSELoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    loss = loss_fcn(input, target)

    optimiser.zero_grad()
    output.backward()
    optimiser.step()
```

# Useful Resources

This is a running list of resources that I considered useful in my DRL journey. 
- [Making sense of the Bias-Variance trade-off](https://blog.mlreview.com/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565) 
- 

