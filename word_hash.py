import numpy as np
from collections import Counter, defaultdict
from blingfire import text_to_words
from tqdm import tqdm
from itertools import product

def tokenizer(string):
    return text_to_words(string).split(' ')


class CharIdf:
    def __init__(self, all_letters, ngrams=3, tokenizer=tokenizer):
        self.ngrams = ngrams
        self.all_letters = list(set(all_letters))
        self.grams = []
        for n in range(1, self.ngrams + 1):
            self.grams += ["".join(i) for i in product(self.all_letters, repeat=n)]
        self.gram_length = len(self.grams)
        self.gram_to_index = {gram: index for index, gram in enumerate(self.grams)}
        self.tokenizer = tokenizer
        self.cache = {}

    def _make_grams(self, word):
        "Make char n-grams from words"
        # skip those chars which you know nothing about
        word = [i for i in word if i in self.all_letters]
        for i in range(len(word)):
            for j in range(i, i + self.ngrams):
                yield "".join(word[i : j + 1])

    def __getitem__(self, word):
        "Get a word's vector"
        if word not in self.cache:
            vec = np.zeros(self.gram_length)
            for gram in self._make_grams(word):
                if gram in self.gram_to_index:
                    vec[self.gram_to_index[gram]] += 1
            self.cache[word] = vec
        return self.cache[word] / self.idf

    def fit(self, docs):
        "Learn idfs"
        self.idf = np.ones(self.gram_length)
        self.cache = {}
        for doc in docs:
            for word in set(self.tokenizer(doc)):
                for gram in self._make_grams(word):
                    self.idf[self.gram_to_index[gram]] += 1

    def transform(self, docs):
        "Get vectors for list of strings"
        docvecs = np.zeros((len(docs), self.gram_length))
        for index, doc in enumerate(tqdm(docs)):
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
