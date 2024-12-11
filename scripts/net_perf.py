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
net_types = ['embedding', 'erdosrenyi', 'disk', 'random']  # Match lowercase conventions for consistency
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
        return dict(type='erdosrenyi', p=0.9 / (n_agents - 1))
    elif nt == 'embedding':
        return dict(type='embedding',
                    duration=ss.constant(0),
                    participation=ss.bernoulli(p=1),
                    debut=ss.constant(0),
                    rel_part_rates=1.0)
    elif nt == 'random':
        return dict(type='random',
                    n_contacts=ss.constant(10))
    elif nt == 'mf':
        return dict(type='mf',
                    duration=ss.constant(0),
                    participation=ss.bernoulli(p=1),
                    debut=ss.normal(loc=0),
                    rel_part_rates=1.0)
    else:
        raise ValueError(f'Unknown network type: {net_type}')

def run_network(n_agents, n_steps, network, rand_seed, rng, idx):
    sim = ss.Sim(pars=dict(
        n_agents=n_agents,
        networks=network,
        label=f'{network["type"]}_N{n_agents}_S{rand_seed}'
    ))
    sim.init()

    network_obj = sim.networks['networks'][0]
    timer = sc.timer()
    for _ in range(n_steps):
        network_obj.step()
    avg_time_per_step = timer.toc(output=True) / n_steps

    return avg_time_per_step, n_agents, network['type'], rand_seed, rng, idx

results = []
for rng in rngs:
    ss.options(rng=rng)
    configs = []
    for net_type in net_types:
        for n in n_agents:
            net = get_net(net_type, n)
            if n >= 50_000 and net_type == 'embedding':
                continue  # Skip embedding over 50k agents
            for seed in range(n_seeds):
                configs.append({
                    'n_agents': n,
                    'n_steps': n_steps,
                    'network': net,
                    'rand_seed': seed,
                    'rng': rng,
                    'idx': len(configs)
                })

    results.extend(sc.parallelize(run_network, iterkwargs=configs, die=True))

# Process results
columns = ['Time per Step', 'Num Agents', 'Network', 'Seed', 'Generator', 'Index']
df = pd.DataFrame(results, columns=columns)
df.to_csv(os.path.join(figdir, 'perf.csv'))

dfm = pd.melt(df, id_vars=['Network', 'Num Agents', 'Generator'], var_name='Channel', value_name='Result')

# Plotting
col_order = ['Time per Step']
g = sns.relplot(kind='line', data=dfm, col='Channel', col_order=col_order, hue='Network', style='Generator',
                x='Num Agents', y='Result', facet_kws={'sharey': False}, height=3, aspect=1.7)
g.set(xscale='log', yscale='log')
sc.savefig(os.path.join(figdir, 'perf.pdf'), g.figure, bbox_inches='tight', transparent=True)
