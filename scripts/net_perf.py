
# Network performance scaling, used to produce Figure C2

import starsim as ss
import numpy as np
import pandas as pd
import seaborn as sns
import sciris as sc
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

# Avoid numpy multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

n_steps = 5
n_seeds = 3
net_types = ['Embedding', 'ErdosRenyi', 'Disk', 'Random'] # 'MF'
rngs = ['multi']
n_agents = np.logspace(1, np.log10(100_000), 9)[:-1]

basedir = os.path.join(os.getcwd(), 'figs')
figdir = os.path.join(basedir, 'Net_perf')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def get_net(net_type, n_agents):
    nt = net_type.lower()
    if nt == 'disk':
        return dict(type='disk', r=0.2, v=0.2 * 365)

    elif nt == 'erdosrenyi':
        return dict(type='erdosrenyi', p = 0.9/(n_agents-1))

    elif nt == 'embedding':
        return dict(type='embedding', 
                duration = ss.constant(0),
                participation = ss.bernoulli(p=1),
                debut = ss.constant(v=0),
                rel_part_rates = 1.0
            )

    elif nt == 'random':
        return dict(type='random', 
                n_contacts = ss.constant(10),
            )

    elif nt == 'mf':
        return dict(type='mf', 
                duration = ss.constant(0),
                participation = ss.bernoulli(p=1),
                debut = ss.normal(loc=0),
                rel_part_rates = 1.0,
            )

    raise Exception(f'Unknown net {net_type}')


def run_network(n_agents, n_steps, network, rand_seed, rng, idx):
    sim = ss.Sim(pars=dict(n_agents=n_agents, networks=network, label=f'{network}_N{n_agents}_S{rand_seed}'))
    sim.init()

    n = sim.networks()[0]
    T = sc.timer()
    for i in range(n_steps):
        n.update()
    dt = T.toc(output=True) / n_steps

    return (dt, n_agents, network['type'], rand_seed, rng, idx)


results = []
for rng in rngs:
    ss.options(_centralized = rng=='centralized')
    cfgs = []
    for net_type in net_types:
        for n in n_agents:
            net = get_net(net_type, n)

            seeds = n_seeds
            steps = n_steps
            if n >= 50_000:
                if net_type == 'Embedding':
                    continue # Skip Embedding over 50k agents

            for rs in range(seeds):
                cfgs.append({'n_agents':n, 'n_steps':n_steps, 'network':net, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})
    results += sc.parallelize(run_network, iterkwargs=cfgs, die=True, serial=True) # , maxmem=0.8

df = pd.DataFrame(results, columns=['Time per Step', 'Num Agents', 'Network', 'Seed', 'Generator', 'Index'])
df.to_csv(os.path.join(figdir, 'perf.csv'))

dfm = pd.melt(df, id_vars=['Network', 'Num Agents', 'Generator'], var_name='Channel', value_name='Result')

col_order = ['Time per Step']
g = sns.relplot(kind='line', data=dfm, col='Channel', col_order=col_order, hue='Network', style='Generator', x='Num Agents', y='Result', facet_kws={'sharey':False}, height=3, aspect=1.7, marker='o')
g.set(xscale='log', yscale='log')

sc.savefig(os.path.join(figdir, 'perf.pdf'), g.figure, bbox_inches='tight', transparent=True)
