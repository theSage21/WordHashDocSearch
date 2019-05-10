import numpy as np
import pandas as pd
from squad_df import v2
from sklearn.feature_extraction.text import TfidfVectorizer
from word_hash import CharIdf
from collections import Counter
from blingfire import text_to_words


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
all_text = "".join(text_to_words(" ".join(df.context).lower()))
all_letters = [letter for letter, count in Counter(all_text).most_common(50)]
print(all_letters)
print(f"{len(all_letters)} letters")

vec = CharIdf(all_letters)
docs = list(set(df.context))
expected_indices = np.array([docs.index(doc) for doc in df.context])
x = vec.fit_transform(docs)
qv = vec.transform(df.question)
result = pd.np.argmax(pd.np.einsum("kd,md->km", x, qv), axis=0)
print("documents", x.shape)
print("questions", qv.shape)
print("matches  ", result.shape)
print("correct  ", (result == expected_indices).mean())
print("diff     ", (expected_indices - result).mean())
