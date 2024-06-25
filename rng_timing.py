import numpy as np
import sciris as sc
import pandas as pd
from starsim.utils import combine_rands

np.random.seed(0)

n = int(1e5) # Nodes
n_edges = 3*n
p = 0.3 # Trans prob

# Create edges
src = np.random.choice(np.arange(n), size=n_edges, replace=True)
trg = np.random.choice(np.arange(n), size=n_edges, replace=True)

# Avoid self edges
keep = src != trg
src = src[keep]
trg = trg[keep]
n_edges = len(src)
prb = np.full(src.shape, fill_value=p)

times = []

# Random
T = sc.tic()
tx_inds = np.random.rand(n_edges) < prb

# Avoid duplicates, pick first source
#new_cases, p2idx, p2inv, p2cnt = np.unique(trg[tx_inds], return_index=True, return_inverse=True, return_counts=True)
#sources = src[tx_inds[p2idx]] # Take first

# Forget about sources
new_cases = trg[tx_inds]
sources = None

times.append( ('Random', sc.toc(T, doprint=False, output=True)) )

# Weyl Middle Square
T = sc.tic()
r1 = np.random.randint(low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64, size=n)
r2 = np.random.randint(low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64, size=n)
tx_inds = np.argwhere(combine_rands(r1[src], r2[trg]) < prb).flatten()

# Avoid duplicates, pick first source
#new_cases, p2idx, p2inv, p2cnt = np.unique(trg[tx_inds], return_index=True, return_inverse=True, return_counts=True)
#sources = src[tx_inds[p2idx]] # Take first

# Forget about sources
new_cases = trg[tx_inds]
sources = None

times.append( ('Weyl Middle Square', sc.toc(T, doprint=False, output=True)) )

# Combine Probabilities
T = sc.tic()
r = np.random.random(size=n)

if True: # Don't care about source
    # Compute node acquisition probability
    node_acq_prob = [1 - np.prod(1-prb[np.argwhere(trg == t).flatten()]) for t in np.arange(n)]
    new_cases = np.argwhere(node_acq_prob < r)
    sources = None
else:
    # Properly address source
    q = np.random.random(size=n)
    p2uniq, p2idx, p2inv, p2cnt = np.unique(trg, return_index=True, return_inverse=True, return_counts=True)

    degrees = np.unique(p2cnt)
    new_cases = []
    sources = []
    for deg in degrees:
        if deg == 1:
            # p2 UIDs that only appear once
            cnt1 = p2cnt == 1
            uids = p2uniq[cnt1] # p2 uids that only have one possible p1 infector
            idx = p2idx[cnt1] # get original index, dfp2[idx] should match uids (above)
            cases = r[uids] < p # determine if p2 acquires from p1
            if cases.any():
                s = src[idx][cases] # Only one possible source for each case because degree is one
        else:
            # p2 UIDs that appear degree times
            dups = np.argwhere(p2cnt==deg).flatten()
            uids = p2uniq[dups]
            
            # Get indices where p2 is repeated
            #  dfp2[inds] should have degree repeated values on each row
            #  dfp1[inds] and dfp[inds] will contain the corresponding sources and transmission probabilities, respectively
            inds = [np.argwhere(np.isin(p2inv, d)).flatten() for d in dups]
            probs = prb[inds]
            p_acq_node = 1-np.prod(1-probs, axis=1) # Compute the overall acquisition probability for each susceptible node that is at risk of acquisition

            cases = r[uids] < p_acq_node # determine if p2 acquires from any of the possible p1s
            if cases.any():
                # Vectorized roulette wheel to pick from amongst the possible sources for each new case, with probability weighting
                # First form the normalized cumsum
                cumsum = probs[cases].cumsum(axis=1)
                cumsum /= cumsum[:,-1][:,np.newaxis]
                # Use slotted q to make the choice
                ix = np.argmax(cumsum >= q[uids[cases]][:,np.newaxis], axis=1)
                # And finally identify the source uid
                s = np.take_along_axis(src[inds][cases], ix[:,np.newaxis], axis=1).flatten()#dfp1[inds][cases][np.arange(len(cases)),ix]

        if cases.any():
            new_cases.append(uids[cases])
            sources.append(s)

    new_cases = np.concatenate(new_cases)
    sources = np.concatenate(sources)

times.append( ('Combine Probabilities', sc.toc(T, doprint=False, output=True)) )

tdf = pd.DataFrame(times, columns=['Method', 'Time'])

print(tdf)

