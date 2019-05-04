import numpy as np
from collections import Counter, defaultdict
from blingfire import text_to_words
from tqdm import tqdm
from itertools import permutations


class CharIdf:
    def __init__(self, all_letters, ngrams=3, tokenizer=text_to_words):
        self.ngrams = ngrams
        self.all_letters = list(set(all_letters))
        self.grams = []
        for i in range(1, ngrams + 1):
            self.grams += ["".join(i) for i in permutations(self.all_letters, i)]
        self.gram_length = len(self.grams)
        self.gram_to_index = {gram: index for index, gram in enumerate(self.grams)}
        self.tokenizer = tokenizer

    def _make_grams(self, word):
        "Make char n-grams from words"
        # skip those chars which you know nothing about
        word = [i for i in word if i in self.all_letters]
        for i in range(len(word)):
            for j in range(i, i + self.ngrams):
                yield "".join(word[i : j + 1])

    def __getitem__(self, word, cache={}):
        "Get a word's vector"
        if word not in cache:
            vec = np.zeros(self.gram_length)
            for gram in self._make_grams(word):
                if gram in self.gram_to_index:
                    vec[self.gram_to_index[gram]] += 1
            cache[word] = vec
        return cache[word]

    def fit(self, docs):
        "Learn idfs"
        self.idf = defaultdict(int)
        for doc in docs:
            for word in set(self.tokenizer(doc)):
                self.idf[word] += 1

    def transform(self, docs):
        "Get vectors for list of strings"
        docvecs = np.zeros((len(docs), self.gram_length))
        for index, doc in enumerate(tqdm(docs)):
            for word, count in Counter(self.tokenizer(doc)).items():
                v = (self[word] * count) / (1 + self.idf[word])
                docvecs[index] += v
        return docvecs

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)

if __name__ == '__main__':
    vec = CharIdf()
    docs = list(set(df.context))
    x = vec.fit_transform(docs)
    qv = vec.transform(df.question)
    result = np.einsum("kd,md->km", x, qv)
