import pickle
import random

from tensorflow.keras.layers import TextVectorization

##load normalized sentence pairs

with open("text_pairs.pickle", "rb") as fp:
    text_pair = pickle.load(fp)

random.shuffle(text_pair)
n_val =  int(0.14*len(text_pair))
n_train = len(text_pair) - 2* n_val
train_pairs = text_pair[:n_train]
val_pairs = text_pair[n_train:n_train+n_val]
test_pairs = text_pair[n_train+n_val:]


vocab_size_en = 10000
vocab_size_fre = 20000
seq_length = 20


eng_vectorizer = TextVectorization(
    max_tokens=vocab_size_en, #max length of vocab for this layer
    standardize=None, #None, lower_strip_punctuation (Laready done)
    split="whitespace", #none, whitespace, character 
    output_mode="int",
    #int, multi_hotm count, tf_idf Int = one integer per split string token (o for masked)
    # multi_hot = outputs a single int array per batch
    #count = like multi_hot but int array contatins a count of the number of times the token at theat index appeared in the batch item
    #tf_idf = like multi_hot but tf_idf alogi is appled to find the valuye in each token slot. 
    output_sequence_length=seq_length,
)
fre_vectorizer = TextVectorization(
    max_tokens=vocab_size_fre,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length + 1
)


train_eng_texts = [pair[0] for pair in train_pairs]
train_fre_texts = [pair[1] for pair in train_pairs]
eng_vectorizer.adapt(train_eng_texts)
fre_vectorizer.adapt(train_fre_texts)

with open("vectoization.pickle", "wb") as fp:
    data = {
        "train" : train_pairs,
        "val" :val_pairs, 
        "test": test_pairs,
        "engvec_config" : eng_vectorizer.get_config(),
        "frevec_config": fre_vectorizer.get_config(),
        "engvec_weights": eng_vectorizer.get_weights(),
        "frevec_weights": fre_vectorizer.get_weights(),


    }
    pickle.dump(data, fp)