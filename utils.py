import gym 
import numpy as np
import matplotlib.pyplot as plt
        
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
        
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action


def plot(frame_idx, rewards):
    plt.figure(1, figsize=(5,5))
    plt.clf()
    plt.title(f"Frame: {frame_idx}, Reward: {rewards[-1]}")
    plt.plot(rewards)
    plt.pause(0.00001) 

def plot_final(frame_idx, rewards, algorithm):
    plt.figure(1, figsize=(5,5))
    plt.clf()
    plt.title(algorithm)
    plt.plot(rewards)
    plt.grid(True)
    plt.savefig(f"images/{algorithm}.png")
        
colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
        
my_red = "#e60049"
my_blue = "#0bb4ff"
my_green = "#50e991"


