import numpy as np
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from squad_df import v2
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from word_hash import CharIdf
from collections import Counter
from blingfire import text_to_words
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def measure(part):
    all_text = "".join(text_to_words(" ".join(part.context).lower()))
    all_letters = [letter for letter, count in Counter(all_text).most_common(50)]
    vec = CharIdf(all_letters)
    docs = list(set(part.context))
    expected_indices = np.array([docs.index(doc) for doc in part.context])
    x = vec.fit_transform(docs)
    qv = vec.transform(part.question)
    result = linear_kernel(x, qv)
    result = pd.np.argmax(result, axis=0)
    char_correct = (result == expected_indices).mean()
    # -------------
    vec = TfidfVectorizer()
    docs = list(set(part.context))
    expected_indices = np.array([docs.index(doc) for doc in part.context])
    x = vec.fit_transform(docs)
    qv = vec.transform(part.question)
    result = linear_kernel(x, qv)
    result = pd.np.argmax(result, axis=0)
    tfidf_correct = (result == expected_indices).mean()
    return char_correct, tfidf_correct

df = pd.DataFrame(list(v2))
df = df.loc[df.is_train]
char_correct = []
tfidf_correct = []
samples = []
bootstrap = 100
for sample_size in [10, 50, 100, 150, 200, 300, 600, 1000]:
    args = []
    for bootstrap in tqdm(range(bootstrap), desc='sampling'):
        part = df.sample(df.shape[0])
        part = part.reset_index()
        part = part[:sample_size]
        args.append(part.copy())
    samples.append(sample_size)
    with Pool() as pool:
        work = pool.imap_unordered(measure, args)
        for a, b in tqdm(work,total=len(args), desc='Work'):
            char_correct.append(a)
            tfidf_correct.append(b)

df = pd.DataFrame({'char': char_correct, 'tfidf': tfidf_correct, 'samples': samples})
df.to_csv('results.csv')
