# Perform tests of pairwise random numbers as presented in Appendix A
# WARNING: This script can take a while to run

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.util import hash_pandas_object
import hashlib
import sciris as sc
import os
import scipy.stats as sps

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

from starsim.utils import combine_rands

import warnings
warnings.filterwarnings("ignore", "overflow encountered in scalar subtract")
warnings.filterwarnings("ignore", "overflow encountered in scalar multiply")

np.random.seed(0)  # Shouldn't matter, but for reproducibility

n = 4  # Number of nodes, 4 or 6

reps = [100_000, 2_000_000][1]
edge_prob = 0.5  # Edge probability

figdir = os.path.join(os.getcwd(), 'figs', f'ERCorr_n{n}_reps{reps}')
os.makedirs(figdir, exist_ok=True)  # Corrected sc.path to os.makedirs


def hash(df):
    return hashlib.sha256(hash_pandas_object(df, index=True).values).hexdigest()[:6]


def random():
    n1, n2 = np.triu_indices(n=n, k=1)
    r = np.random.rand(len(n1))
    edge = r < edge_prob
    return n1[edge], n2[edge]


def modulo():
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)

    n1, n2 = np.triu_indices(n=n, k=1)
    r = np.mod(r1[n1] + r2[n2], 1)
    edge = r < edge_prob
    return n1[edge], n2[edge]


def middle_sq():
    r1 = np.random.randint(low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64, size=n)
    r2 = np.random.randint(low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64, size=n)

    n1, n2 = np.triu_indices(n=n, k=1)

    a = r1[n1]
    b = r2[n2]
    x = a * b
    y = x.copy()
    z = y + b

    # Round 1
    x = x * x + y
    x = np.bitwise_or(np.right_shift(x, 32), np.left_shift(x, 32))

    # Round 2
    x = x * x + z
    x = np.bitwise_or(np.right_shift(x, 32), np.left_shift(x, 32))

    # Round 3
    x = x * x + y
    x = np.bitwise_or(np.right_shift(x, 32), np.left_shift(x, 32))

    # Round 4
    x = x * x + z
    t = x.copy()
    x = np.bitwise_or(np.right_shift(x, 32), np.left_shift(x, 32))

    # Round 5 and xor
    x = np.right_shift(x * x + y, 32)
    r = np.right_shift(t, x)  # Corrected bitwise xor operation to `right_shift`

    edge = r < edge_prob

    return n1[edge], n2[edge]


g = np.random.default_rng()  # Updated to use numpy's default RNG
def xor():
    r1 = g.integers(0, 2**32 - 1, size=n, dtype=np.uint32)
    r2 = g.integers(0, 2**32 - 1, size=n, dtype=np.uint32)

    n1, n2 = np.triu_indices(n=n, k=1)
    r = combine_rands(r1[n1], r2[n2])
    edge = r < edge_prob
    return n1[edge], n2[edge]


def make_graph(trans_fn):
    els = {}
    counts = {}

    T = sc.tic()
    for _ in range(reps):
        p1, p2 = trans_fn()
        df = pd.DataFrame({'p1': p1, 'p2': p2}).sort_values(['p1', 'p2']).reset_index(drop=True)

        h = hash(df)

        if h not in els:
            els[h] = df

        counts[h] = counts.get(h, 0) + 1
    dt = sc.toc(T, doprint=False, output=True)

    df = pd.DataFrame(counts.values(), index=pd.Index(counts.keys(), name='Graph Hash'), columns=['Counts'])
    return els, df, dt


if __name__ == '__main__':
    results = [make_graph(fn) for fn in [random, modulo, middle_sq, xor]]
    tx, cnt, times = zip(*results)

    df = pd.concat(cnt, axis=1) \
        .fillna(0) \
        .astype(int)
    df.columns = [
        'True Random',
        'Modulo',
        'Middle Square',
        'XOR'
    ]

    df.reset_index(inplace=True)
    df.to_csv(os.path.join(figdir, 'results.csv'))

    dfm = df.melt(id_vars='Graph Hash', var_name='Method', value_name='Count')

    chisq = []
    f_exp = df['True Random']
    for method in df.columns[2:]:
        f_obs = df[method]
        chisq.append((method, sps.chi2_contingency([f_exp, f_obs]).pvalue))
    x2df = pd.DataFrame(chisq, columns=['Method', 'P-Value']).set_index('Method')
    x2df.to_csv(os.path.join(figdir, 'chisq.csv'))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dfm, x='Graph Hash', y='Count', hue='Method', ax=ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, 'graph_hist.pdf'), bbox_inches='tight', transparent=True)
    plt.close()

    txc = tx[0].copy()
    for i in range(1, len(tx)):
        txc.update(tx[i])
    for h, v in txc.items():
        print(f'\nUnique graph #{h}')
        print(v)

    print(times)
