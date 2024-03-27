import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_scenarios(df, figdir, channels=None):
    sns.set(font_scale=1.4, style='whitegrid')

    # Renaming
    df.replace({'rng': {'centralized':'Centralized', 'multi': 'CRN'}}, inplace=True)
    rngs = ['Centralized', 'CRN']

    covs = df['cov'].unique()
    covs.sort()
    rep = {c: f'{c:.0%}' if c > 0 else 'Reference' for c in covs}
    df.replace({'cov': rep}, inplace=True)
    cov_ord = [rep[c] for c in covs]
    first = cov_ord.pop(0)
    cov_ord.append(first)
    df['cov'] = pd.Categorical(df['cov'], categories=cov_ord) # Agh!

    df.rename(columns={'cov': 'Coverage'}, inplace=True)
    cov = 'Coverage'

    d = pd.melt(df, id_vars=['year', 'rand_seed', cov, 'rng', 'network'], var_name='channel', value_name='Value')
    d['baseline'] = d[cov]=='Reference'
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]
    bl = bl.set_index(['year', 'channel', 'rand_seed', cov, 'rng', 'network'])[['Value']].reset_index(cov)
    scn = scn.set_index(['year', 'channel', 'rand_seed', cov, 'rng', 'network'])[['Value']].reset_index(cov)
    mrg = scn.merge(bl, on=['year', 'channel', 'rand_seed', 'rng', 'network'], suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg = mrg.sort_index()
    mrg['Coverage'] = pd.Categorical(mrg['Coverage'], categories=cov_ord[:-1]) # Agh^2!

    cor = mrg.groupby(['year', 'channel', 'rng', cov])[['Value', 'Value_ref']].apply(lambda x: np.corrcoef(x['Value'], x['Value_ref'], rowvar=False)[0,1])
    cor.name = 'Pearson'
    #cor.replace([np.inf, -np.inf, np.nan], 1, inplace=True)

    kw = {'height': 3, 'aspect': 1.4}
    fkw = {'sharey': False, 'sharex': 'col', 'margin_titles': True}

    # Make a color palette for timeseries
    Set1_mod = sns.color_palette('Set1') 
    #Set1_mod = [Set1_mod[2]] + [Set1_mod[0]] + [Set1_mod[1]]
    #print(Set1_mod)
    first = Set1_mod.pop(0) # Move the 1st color to len(cov) for consistency
    n = len(covs)-1
    Set1_mod = Set1_mod[:n] + [first]
    #print(Set1_mod)

    ## TIMESERIES
    g = sns.relplot(kind='line', data=d, x='year', y='Value', hue=cov, hue_order=cov_ord, col='channel', col_order=channels, row='rng', row_order=rngs,
        palette='Set1', errorbar='sd', lw=2, facet_kws=fkw, **kw)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.set_xlabels('Year')
    g.figure.savefig(os.path.join(figdir, 'timeseries.png'), bbox_inches='tight', dpi=300)

    ## TIMESERIES: specific channels
    for ch in ['Maternal Deaths', 'Infected']:
        if ch in d['channel'].unique():
            g = sns.relplot(kind='line', data=d, x='year', y='Value', hue=cov, hue_order=cov_ord, row='channel', row_order=[ch], col='rng', col_order=rngs,
                palette=Set1_mod, errorbar='sd', lw=2, facet_kws=fkw, **kw)
            g.set_titles(col_template='{col_name}', row_template='{row_name}')
            g.set_xlabels('Year')
            g.figure.savefig(os.path.join(figdir, f'timeseries_{ch.replace(" ", "")}.png'), bbox_inches='tight', dpi=300)

    ## DIFF TIMESERIES
    for ms, mrg_by_ms in mrg.groupby('rng'):
        g = sns.relplot(kind='line', data=mrg_by_ms, x='year', y='Value - Reference', hue=cov, col='channel', col_order=channels,
                palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw, **kw) # row=cov, 
        g.set_titles(col_template='{col_name}', row_template='{row_name}')
        #g.figure.suptitle('MultiRNG' if ms else 'SingleRNG')
        g.figure.subplots_adjust(top=0.88)
        g.set_xlabels('Year')
        g.figure.savefig(os.path.join(figdir, f'diff_{ms}.png'), bbox_inches='tight', dpi=300)

    ## FINAL TIME
    tf = df['year'].max()
    mtf = mrg.loc[tf]
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue=cov, x='Value - Reference',
            col='channel', col_order=channels, row='rng', row_order=rngs, facet_kws=fkw, palette='Set1', **kw)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.set_xlabels(f'Value - Reference at year {tf}')
    g.figure.savefig(os.path.join(figdir, 'final.png'), bbox_inches='tight', dpi=300)

    ## COR SCATTER FINAL TIME
    ctf = mtf.reset_index('rand_seed').set_index(cov, append=True).sort_index()
    g = sns.relplot(data=ctf, kind='scatter', hue='rng', hue_order=rngs, style='rng', style_order=rngs, x='Value_ref', y='Value',
            col='channel', col_order=channels, row=cov, facet_kws=fkw, palette='tab10', **kw)
    g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
    g.set_xlabels(f'Reference at year {tf}')
    g.set_ylabels(f'Value at year {tf}')
    g.figure.savefig(os.path.join(figdir, 'cor_final.png'), bbox_inches='tight', dpi=300)

    # Share y from here on
    fkw['sharey'] = 'row'

    ## DIFF TIMESERIES ALL IN ONE
    g = sns.relplot(kind='line', data=mrg, x='year', y='Value - Reference', hue=cov, row='channel', row_order=channels,
            col='rng', col_order=rngs, palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw, **kw)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.figure.subplots_adjust(top=0.88)
    g.set_xlabels('Year')
    g.figure.savefig(os.path.join(figdir, f'diff.png'), bbox_inches='tight', dpi=300)


    ## CORRELATION OVER TIME
    try:
        g = sns.relplot(kind='line', data=cor.to_frame(), x='year', y='Pearson', hue='rng', hue_order=rngs, col='channel', col_order=channels,
                row=cov, palette='tab10', errorbar='sd', lw=2, facet_kws=fkw, **kw)
        g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
        g.figure.subplots_adjust(top=0.88)
        g.set_xlabels('Year')
        g.figure.savefig(os.path.join(figdir, 'cor.png'), bbox_inches='tight', dpi=300)
    except:
        print('CORRELATION OVER TIME did not work')

    print('Figures saved to:', os.path.join(os.getcwd(), figdir))

    return



def plot_graph(G):
    # Code based on https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G, seed=10396954)
    nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Graph")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    return fig
