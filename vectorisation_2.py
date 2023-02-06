import pickle
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

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
                "decoder_input":fre[:, :-1]}
    target = fre[:, 1:]
    return (source, target)

def make_dataset(pairs, batch_size = 64):
    #Creating tensorflow dataset for the sentence pairs

    eng_text, fre_text = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(eng_text), list(fre_text)))
    return dataset.shuffle(2048) \
                  .batch(batch_size).map(format_dataset) \
                  .prefetch(16).cache()
"""
This code returns a tf.data.Dataset object that is processed in a particular way. The methods used to process the dataset are:

    .shuffle(2048): shuffles the elements of the dataset randomly. The argument 2048 determines the buffer size to use for shuffling, meaning that the dataset will sample elements from the buffer of size 2048 to create the new dataset.

    .batch(batch_size): groups the elements of the dataset into batches of size batch_size. Each batch will contain batch_size consecutive elements from the dataset.

    .map(format_dataset): applies a function format_dataset to each element of the dataset, which can be used to transform the elements of the dataset in some way. The exact format of the output depends on the implementation of the format_dataset function.

    .prefetch(16): adds a buffer to the dataset to prefetch data for faster consumption. The argument 16 determines the size of the buffer, meaning that the dataset will buffer 16 elements ahead of the current position for faster retrieval.

    .cache(): caches the elements of the dataset so that they do not need to be processed again when the same dataset is consumed multiple times.

The overall effect of these operations is to create a processed dataset that can be efficiently consumed in a batch-wise manner, with elements that are randomly shuffled, transformed by the format_dataset function, prefetched, and cached for faster access.
"""
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
 
# test the dataset
"""for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["encoder_inputs"][0]: {inputs["encoder_inputs"][0]}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"][0]: {inputs["decoder_inputs"][0]}')
    print(f"targets.shape: {targets.shape}")
    print(f"targets[0]: {targets[0]}")"""