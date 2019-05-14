import numpy as np
from tqdm import tqdm
from itertools import product
from blingfire import text_to_words
from collections import Counter, defaultdict


def tokenizer(string):
    return text_to_words(string).split(' ')


class CharIdf:
    def __init__(self, all_letters, ngrams=(2, 4), tokenizer=tokenizer, verbose=False):
        self.ngrams = ngrams
        self.all_letters = list(set(all_letters))
        self.grams = []
        for n in range(self.ngrams[0], self.ngrams[1] + 1):
            self.grams += ["".join(i) for i in product(self.all_letters, repeat=n)]
        self.gram_length = len(self.grams)
        self.gram_to_index = {gram: index for index, gram in enumerate(self.grams)}
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.epsilon = 1e-3

    def _make_grams(self, word):
        "Make char n-grams from words"
        # skip those chars which you know nothing about
        word = [i for i in word if i in self.all_letters]
        seen = set()
        for i in range(len(word)):
            for j in range(i+self.ngrams[0], i + self.ngrams[1]):
                w = "".join(word[i : j + 1])
                if w not in seen and (self.ngrams[0] <= len(w) <= self.ngrams[1]):
                    seen.add(w)
                    yield w

    def __getitem__(self, word):
        "Get a word's vector"
        vec = np.zeros(self.gram_length)
        for gram in self._make_grams(word):
            if gram in self.gram_to_index:
                vec[self.gram_to_index[gram]] += len(gram)
        return vec / self.idf

    def fit(self, docs):
        "Learn idfs"
        self.idf = np.ones(self.gram_length) * self.epsilon
        for doc in docs:
            for word in set(self.tokenizer(doc)):
                for gram in self._make_grams(word):
                    self.idf[self.gram_to_index[gram]] += len(gram)

    def transform(self, docs):
        "Get vectors for list of strings"
        docvecs = np.zeros((len(docs), self.gram_length))
        docs = tqdm(docs) if self.verbose else docs
        for index, doc in enumerate(docs):
            for word, count in Counter(self.tokenizer(doc)).items():
                docvecs[index] += self[word]
        return docvecs

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)


if __name__ == "__main__":
    vec = CharIdf()
    docs = list(set(df.context))
    x = vec.fit_transform(docs)
    qv = vec.transform(df.question)
    result = np.einsum("kd,md->km", x, qv)
