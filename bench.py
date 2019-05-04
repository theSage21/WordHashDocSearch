import pandas as pd
from squad_df import v2
from word_hash import CharIdf
from collections import Counter


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

vec = CharIdf(all_letters)
docs = list(set(df.context))
x = vec.fit_transform(docs)
qv = vec.transform(df.question)
result = pd.np.argmax(pd.np.einsum("kd,md->km", x, qv), axis=1)
print('documents', x.shape)
print('questions', qv.shape)
print('matches  ', result.shape)
