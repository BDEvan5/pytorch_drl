import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf

# Define the network architecture with the Flax Linen API
class MNISTNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x
# class MNISTNet(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(x, features=256)
#         x = nn.relu(x)
#         x = nn.Dense(x, features=10)
#         return x

# Define the loss function and metrics
def cross_entropy_loss(logits, labels):
    return jnp.mean(jnp.sum(jnp.square(logits - labels), axis=1))

def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=1) == labels)

# Load the MNIST dataset
# train_data, test_data = tfds.load("mnist", split=["train", "test"])

train_data = tfds.load('mnist', split='train')
test_data = tfds.load('mnist', split='test')
train_data = train_data.map(lambda sample: {'image': tf.cast(sample['image'],
                                                        tf.float32) / 255.,
                                        'label': sample['label']}) # normalize train set
test_data = test_data.map(lambda sample: {'image': tf.cast(sample['image'],
                                                        tf.float32) / 255.,
                                    'label': sample['label']}) # normalize test set
# Preprocess the data
def preprocess(x, y):
    x = jnp.array(x, dtype=jnp.float32) / 255.0
    y = jnp.array(y, dtype=jnp.int32)
    return x, flax.deprecated.nn.one_hot(y, num_classes=10)

# train_data = train_data.map(preprocess)
# test_data = test_data.map(preprocess)

# Define the optimizer
learning_rate = 0.001
import optax
optimizer = optax.adam(learning_rate)
# optimizer = flax.optim.Adam(learning_rate=learning_rate)

# Define the training and evaluation functions
@jax.jit
def train_step(params, state, batch):
    logits = MNISTNet().apply(params, batch[0])
    loss, metrics = cross_entropy_loss(logits, batch[1]), accuracy(logits, batch[1])
    grad = jax.grad(loss)(params)
    new_params, state = optimizer.update(grad, state)
    return new_params, state, metrics

@jax.jit
def eval_step(params, batch):
    logits = MNISTNet().apply(params, batch[0])
    loss, metrics = cross_entropy_loss(logits, batch[1]), accuracy(logits, batch[1])
    return metrics

# Initialize model parameters and optimizer state
rng = jax.random.PRNGKey(0)
params = MNISTNet()
# params = MNISTNet().init(rng)
state = optimizer.init(params)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_data:
        params, state, metrics = train_step(params, state, batch)
    # Evaluate the model on the test set
    test_metrics = []
    for batch in test_data:
        test_metrics.append(eval_step(params, batch))
    test_loss, test_accuracy = jnp.mean(test_metrics, axis=0)
    print(f"Epoch: {epoch + 1}, Train loss: {loss}, Train accuracy: {metrics[1]}, Test loss: {test_loss}, Test accuracy: {test_accuracy}")

# Print the final test accuracy
print(f"Final test accuracy: {test_accuracy}")
