import tensorflow as tf
import numpy as np
import time

# Create a large random matrix
print("Creating large tensors...")
a = tf.random.normal([50000, 50000])
b = tf.random.normal([50000, 50000])

# Force execution and timing
print("Running matrix multiplication...")
start = time.time()
c = tf.matmul(a, b)
# Force execution with .numpy()
result = c.numpy()
end = time.time()

print(f"Computation took {end-start:.2f} seconds")
print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")
print(f"Device used: {tf.device('/GPU:0') if tf.config.list_physical_devices('GPU') else tf.device('/CPU:0')}")