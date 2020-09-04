import tensorflow as tf 
import numpy as np 


data = np.reshape(np.arange(40), [20, 2])
indices = np.random.randint(0, 2, [20])[:, None]
index = np.arange(20)[:, None]
indices = np.concatenate([index, indices], axis=-1)

print(data)
print(indices)

data = tf.convert_to_tensor(data, dtype=tf.float32)
indices = tf.convert_to_tensor(indices, dtype=tf.int32)
index_data = tf.gather_nd(data, indices)
# index_data = tf.gather(data, indices)

print(index_data.numpy())



