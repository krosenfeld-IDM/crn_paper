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

n_steps = 5 # Was 25
n_seeds = 3 # was 10
#net_types = ['ErdosRenyi', 'Disk', 'Embedding']
net_types = ['Embedding', 'ErdosRenyi', 'Disk', 'Random'] # 'MF'
#rngs = ['centralized', 'multi']
rngs = ['multi']
#n_agents = np.logspace(1, np.log10(30_000), 20) # np.log10(10000)
n_agents = np.logspace(1, np.log10(100_000), 9)[:-1]
#n_agents = [50_000] # 50k --> 1_249_975_000 possible edges

basedir = os.path.join(os.getcwd(), 'figs')
figdir = os.path.join(basedir, 'Net_perf')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def get_net(net_type, n_agents):
    nt = net_type.lower()
    if nt == 'disk':
        return dict(type='disk', r=0.2, v=0.2 * 365)
        #ss.DiskNet(r=0.075 * (50/n_agents)**0.5, v=0.2 * 365),
        #dict(type='disk', r=0.075, v=0.2 * 365),

    elif nt == 'erdosrenyi':
        return dict(type='erdosrenyi', p = 0.9/(n_agents-1))

    elif nt == 'embedding':
        return dict(type='embedding', 
                duration = ss.constant(0), #ss.lognorm_ex(mean=1),  # Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
                participation = ss.bernoulli(p=1),  # Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
                debut = ss.constant(v=0),  # Age of debut can vary by using callable parameter values
                rel_part_rates = 1.0
            )

    elif nt == 'random':
        return dict(type='random', 
                n_contacts = ss.constant(10),
            )

    elif nt == 'mf':
        return dict(type='mf', 
                duration = ss.constant(0), #ss.lognorm_ex(mean=1),  # Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
                participation = ss.bernoulli(p=1),  # Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
                debut = ss.normal(loc=0),  # Age of debut can vary by using callable parameter values
                rel_part_rates = 1.0,
            )


    raise Exception(f'Unknown net {net_type}')
    return


def run_network(n_agents, n_steps, network, rand_seed, rng, idx):
    sim = ss.Sim(n_agents=n_agents, networks=network, label=f'{network}_N{n_agents}_S{rand_seed}')
    sim.initialize()

    #n_edges = 0
    n = sim.networks[0]
    T = sc.tic()
    for i in range(n_steps):
        n.update()
        #n_edges += len(n.p1)
    dt = sc.toc(T, output=True) / n_steps

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
                ######seeds = n_seeds // 5 # Fewer seeds when many agents
                ######steps = n_steps // 5 # Fewer steps when many agents
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