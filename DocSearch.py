import pandas as pd
import numpy as np
from squad_df import v2
from collections import Counter, defaultdict
from blingfire import text_to_words
from tqdm import tqdm_notebook
from itertools import permutations


def build(df):
    df = df[["question", "context"]]
    passages = list(set(df.context.values))
    ptoi = {p: i for i, p in enumerate(passages)}
    itop = {i: p for p, i in ptoi.items()}
    dataset = []
    for _, row in df.iterrows():
        dataset.append((row.question, ptoi[row.context], row.context))
    return (
        pd.DataFrame(dataset, columns=["question", "ctxid", "relevant"]),
        ptoi,
        itop,
        passages,
    )


df = pd.DataFrame(list(v2))
df = df.loc[df.is_train]
df = df.sample(df.shape[0])
df = df.reset_index()
df = df[:1000]  # keep it small for now
all_letters = [
    letter
    for letter, count in Counter("".join(df.context).lower()).items()
    if count > 1000
]


class CharIdf:
    def __init__(self, ngrams=3):
        self.ngrams = ngrams
        global all_letters
        self.grams = []
        for i in range(1, ngrams + 1):
            self.grams += ["".join(i) for i in permutations(all_letters, i)]
        self.gram_length = len(self.grams)
        self.gram_to_index = {gram: index for index, gram in enumerate(self.grams)}

    def _make_grams(self, word):
        word = list(word)
        for i in range(len(word)):
            for j in range(i, i + self.ngrams):
                yield "".join(word[i : j + 1])

    def __getitem__(self, word, cache={}):
        if word not in cache:
            vec = np.zeros(self.gram_length)
            for gram in self._make_grams(word):
                if gram in self.gram_to_index:
                    vec[self.gram_to_index[gram]] += 1
            cache[word] = vec
        return cache[word]

    def fit(self, docs):
        self.idf = defaultdict(int)
        for doc in docs:
            for word in set(text_to_words(doc)):
                self.idf[word] += 1

    def transform(self, docs):
        docvecs = np.zeros((len(docs), self.gram_length))
        print("making vectors")
        for index, doc in enumerate(tqdm_notebook(docs)):
            for word, count in Counter(text_to_words(doc)).items():
                v = (self[word] * count) / (1 + self.idf[word])
                docvecs[index] += v
        return docvecs

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)


vec = CharIdf()
docs = list(set(df.context))
x = vec.fit_transform(docs)
qv = vec.transform(df.question)


result = np.einsum("kd,md->km", x, qv)
