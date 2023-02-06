#positional encoding layer


import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def positional_ecoding_matrix(L, d, n = 10000):
    assert d % 2 == 0
    d2 = d//2 

    P = np.zeros((L,d))
    k = np.arange(L).reshape(-1,1)
    i = np.arange(d2).reshape(1,-1)
    denom = np.power(n, -i/d2)
    args = k* denom
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P

class PositionalEmbedding(tf.keras.layers.Layer):
    """Positional embedding layer. Assume tokenized input, transform into
    embedding and returns positional-encoded output."""
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        """
        Args:
            sequence_length: Input sequence length
            vocab_size: Input vocab size, for setting up embedding matrix
            embed_dim: Embedding vector size, for setting up embedding matrix
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim     # d_model in paper
        # token embedding layer: Convert integer token to D-dim float vector
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
        )
        # positional embedding layer: a matrix of hard-coded sine values
        matrix = positional_ecoding_matrix(sequence_length, embed_dim)
        self.position_embeddings = tf.constant(matrix, dtype="float32")
 
    def call(self, inputs):
        """Input tokens convert into embedding vectors then superimposed
        with position vectors"""
        embedded_tokens = self.token_embeddings(inputs)
        return embedded_tokens + self.position_embeddings
 
    # this layer is using an Embedding layer, which can take a mask
    # see https://www.tensorflow.org/guide/keras/masking_and_padding#passing_mask_tensors_directly_to_layers
    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args, **kwargs)
 
    def get_config(self):
        # to make save and load a model using custom layer possible
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

vocab_size_en = 10000
seq_length = 20


with open("vectoization.pickle", "rb") as fp:
    data = pickle.load(fp)

train_pairs = data["train"]
test_pairs = data["test"]
val_pairs = data["val"]

 
eng_vectorizer = TextVectorization.from_config(data["engvec_config"])
eng_vectorizer.set_weights(data["engvec_weights"])
fre_vectorizer = TextVectorization.from_config(data["frevec_config"])
fre_vectorizer.set_weights(data["frevec_weights"])



def format_dataset(eng, fre):
    eng = eng_vectorizer(eng)
    fre = fre_vectorizer(fre)
    source = {"encoder_inputs":eng,
                "decoder_inputs":fre[:, :-1]}
    target = fre[:, 1:]
    return (source, target)

def make_dataset(pairs, batch_size = 64):
    #Creating tensorflow dataset for the sentence pairs

    eng_text, fre_text = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(eng_text), list(fre_text)))
    return dataset.shuffle(2048) \
                  .batch(batch_size).map(format_dataset) \
                  .prefetch(16).cache()


train_ds = make_dataset(train_pairs) 

for inputs, targets in train_ds.take(1):
    print(inputs["encoder_inputs"])
    embed_en = PositionalEmbedding(seq_length, vocab_size_en, embed_dim=512)
    en_emb = embed_en(inputs["encoder_inputs"])
    print(en_emb.shape)
    print(en_emb._keras_mask)




"""
Combining embedding layer with postion encoding. The embedding layer creates word embeddings, 
namely, converting an integer token label from the vectorized sentence into a vector that can carry
thi meaning of the word. With the embdding, you can tell how close in meaning the two different words are. 


The embedding ooutput depends on the tokenized input sentence. But the positional encoding is a condatnat
matrix as it depends only on the positioon. hence you create a condatsant tensor for that at the time
you created this layer. ensorflow is smart enough to match the dimensions when you add the smbedding
ouput to the positional matrix, in the call() function. 

Two additional functional are defined in the layer above. The compute_mask() function is passed on to
the embedding layer. This is needed to tell which positions of the ouptut are padded. This
will be used internally by Keras. The get_config() function is defined to remeber all the condif
parameters of this layer. This is a standard practive in keras so that you remember all the para
meters you passed on to the constructor and return them in get_condif(), so the modek can be saved 
and loaded. 
"""









