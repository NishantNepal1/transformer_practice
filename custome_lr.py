import tensorflow as tf
import matplotlib.pyplot as plt
from transformer import transformer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, key_dim, warmup_steps = 4000):
        super().__init__()
        self.key_dim = key_dim
        self.warmup_steps = warmup_steps
        self.d = tf.cast(self.key_dim, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, dtype= tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step *(self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d)*tf.math.minimum(arg1,arg2)
    
    def get_config (self):
        config = {
            "jey_dim": self.key_dim,
            "warmup_steps": self.warmup_steps,

        }
        return config



"""
    lr (learning rate): The step size at which the optimizer makes updates to the model weights. A smaller learning rate means slower updates, while a larger learning rate means faster updates.

    beta_1: The exponential decay rate for the first moment estimates (mean).

    beta_2: The exponential decay rate for the second moment estimates (variance).

    epsilon: A small constant added to the denominator to prevent division by zero.
"""



"""plt.plot(lr(tf.range(50000, dtype = tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()"""

def masked_loss(label, pred):
    mask = label !=0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *=mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

 
def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
 
    mask = label != 0
 
    match = match & mask
 
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

vocab_size_en = 10000
vocab_size_fr = 20000
seq_len = 20
num_layers = 4
num_heads = 8
key_dim = 128
ff_dim = 512
dropout = 0.1
key_dim = 128


model = transformer(num_layers, num_heads, seq_len, key_dim, ff_dim,
                    vocab_size_en, vocab_size_fr, dropout)
lr = CustomSchedule(key_dim)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
model.summary()