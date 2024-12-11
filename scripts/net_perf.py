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
net_types = ['Embedding', 'ErdosRenyi', 'Disk', 'Random']  # 'MF' excluded for now
rngs = ['multi']
n_agents = np.logspace(1, np.log10(100_000), 9, dtype=int)[:-1]

basedir = os.path.join(os.getcwd(), 'figs')
figdir = os.path.join(basedir, 'Net_perf')
os.makedirs(figdir, exist_ok=True)

def get_net(net_type, n_agents):
    nt = net_type.lower()
    if nt == 'disk':
        return ss.DiskNet(pars={'r': 0.2, 'v': 0.2 * 365})

    elif nt == 'erdosrenyi':
        return ss.ErdosRenyiNet(pars={'p': 0.9 / (n_agents - 1)})

    elif nt == 'embedding':
        return ss.EmbeddingNet(pars={
            'duration': ss.dur(0),
            'participation': ss.bernoulli(p=1),
            'debut': ss.dur(0),
            'rel_part_rates': 1.0
        })

    elif nt == 'random':
        return ss.RandomNet(pars={'n_contacts': ss.constant(10)})

    elif nt == 'mf':
        return ss.MFNet(pars={
            'duration': ss.dur(0),
            'participation': ss.bernoulli(p=1),
            'debut': ss.dur(0),
            'rel_part_rates': 1.0
        })

    raise ValueError(f'Unknown network type: {net_type}')

def run_network(n_agents, n_steps, network, rand_seed, rng, idx):
    sim = ss.Sim(
        pars={
            'n_agents': n_agents,
            'networks': [network],
            'label': f'{network.name}_N{n_agents}_S{rand_seed}'
        }
    )
    sim.init()
    sim.t.ti = 0

    network_module = sim.networks[0]
    start_time = sc.tic()
    for _ in range(n_steps):
        network_module.step()
    elapsed_time = sc.toc(start_time, output=True) / n_steps

    return elapsed_time, n_agents, network.name, rand_seed, rng, idx

results = []
for rng in rngs:
    ss.options._centralized = (rng == 'centralized')  # Correctly toggle centralized mode
    config_list = []
    for net_type in net_types:
        for n in n_agents:
            if net_type == 'embedding' and n >= 50_000:
                continue  # Skip embedding networks over 50k agents
            net = get_net(net_type, n)

            for seed in range(n_seeds):
                config_list.append({
                    'n_agents': n,
                    'n_steps': n_steps,
                    'network': net,
                    'rand_seed': seed,
                    'rng': rng,
                    'idx': len(config_list)
                })

    # Run the networks in parallel
    results += sc.parallelize(run_network, iterkwargs=config_list, die=True, serial=True)

# Save results
df = pd.DataFrame(results, columns=['Time per Step', 'Num Agents', 'Network', 'Seed', 'Generator', 'Index'])
df.to_csv(os.path.join(figdir, 'perf.csv'), index=False)

# Plot results
df_melted = pd.melt(df, id_vars=['Network', 'Num Agents', 'Generator'], 
                    var_name='Channel', value_name='Result')

sns.relplot(
    kind='line',
    data=df_melted,
    col='Channel',
    col_order=['Time per Step'],
    hue='Network',
    style='Generator',
    x='Num Agents',
    y='Result',
    facet_kws={'sharey': False},
    height=3,
    aspect=1.7,
    marker='o'
).set(xscale='log', yscale='log')

output_path = os.path.join(figdir, 'perf.pdf')
sc.savefig(output_path, transparent=True, bbox_inches='tight')
