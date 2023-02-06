import pathlib
import pickle
import random
import re
import unicodedata
import matplotlib.pyplot as plt
import tensorflow as tf

text_file = tf.keras.utils.get_file(
    fname = "fra-eng.zip",
    origin = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip",
    extract=True,
)

text_file = pathlib.Path(text_file).parent / "fra.txt" #This will get fra-eng

def normalize(line):
    "Normalize a line of text and split into two at the tab character"
    line = unicodedata.normalize("NFKC", line.strip().lower())
    #NKFC = a standardized form of unicode normalization that ensure that equicalent characters hav
    # a unique representation. 
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
    line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)
    #re.sub(PATTERN, Replcement, String, Flag, count =0, flag = 0)
    eng, fra = line.split("\t")
    #Split by tab
    fra = "[START] "+fra+" [END]"
    return eng, fra

with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]


for _ in range(5):
    print(random.choice(text_pairs))

with open("text_pairs.pickle", "wb") as fp:
    pickle.dump(text_pairs, fp)

with open("text_pairs.pickle", "rb") as fp:
    text_pairs = pickle.load(fp)

#statistics about the imported file 

eng_token, fre_token = set(), set()

eng_max, fre_max = 0, 0 

for eng, fre in text_pairs:
    eng_split = eng.split()
    fre_split = fre.split()

    eng_max = max(eng_max, len(eng_split))
    fre_max = max(fre_max, len(fre_split))

    eng_token.update(eng_split)
    fre_token.update(fre_split)

print("Longest Eng Sentence:  ", eng_max)
print("Longest Fre Sentence:  ", fre_max)
print("Total Eng Words ", len(eng_token))
print("Total Fre Words ", len(fre_token))

# histogram of sentence length in tokens
eng_len = []
fre_len = []
for eng, fre in text_pairs:
    eng_len.append(len(eng.split()))
    fre_len.append(len(fre.split()))



from collections import Counter

#eng_list = Counter((eng_len))
#fre_list = Counter((fre_len))

plt.hist(eng_len, label="en", color="red", alpha=0.33)
plt.hist(fre_len, label="fr", color="blue", alpha=0.33)

plt.yscale("log") 
plt.ylim(plt.ylim()) # make Y-axis consistent for both plots
plt.plot([max(eng_len), max(eng_len)], plt.ylim(), color ="red")
plt.plot([max(fre_len), max(fre_len)], plt.ylim(), color="blue")
plt.legend()
plt.title("Examples count vs Token length")
plt.show()

















