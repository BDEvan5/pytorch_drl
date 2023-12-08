import jax.numpy as jnp 
import jax
import flax.linen as nn
import numpy as np
import optax
import matplotlib.pyplot as plt

data_sz = 10000
xs = np.linspace(0, 10, data_sz)
ys = xs * 2 - 3
ys += np.random.normal(0, 1, data_sz)
n_train = int(0.6 * data_sz)
x_train, y_train = xs[:n_train].reshape(-1, 1), ys[:n_train].reshape(-1, 1)
x_test, y_test = xs[n_train:].reshape(-1, 1), ys[n_train:].reshape(-1, 1)

class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        x = nn.Dense(10)(x)
        return x


epochs = 500
lr = 0.02
model = Network()
rng = jax.random.key(0)
params = model.init(rng, x_train)
params = model.init(rng, jnp.ones((2, 1)))
print(params['params']["Dense_0"]["kernel"].shape)
print(params['params']["Dense_1"]["kernel"].shape)

@jax.jit
def update_params(params, learning_rate, grads):
  params = jax.tree_util.tree_map(
      lambda p, g: p - learning_rate * g, params, grads)
  return params

@jax.jit
def mse_loss(params, x, y):
    pred_y = model.apply(params, x)
    return jnp.mean(jnp.square(pred_y - y))

loss_fcn = jax.value_and_grad(mse_loss)


for e in range(epochs):
    loss_val, grads = loss_fcn(params, x_train, y_train)
    params = update_params(params, lr, grads)

    if e % 20 == 0:
        test_loss, _g = loss_fcn(params, x_test, y_test)
        print(f"Epoch {e} | Loss: {test_loss}")

# print(params)
test_loss, _g = loss_fcn(params, x_test, y_test)
print(f"Final Loss: {test_loss}")

