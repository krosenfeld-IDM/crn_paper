import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import networkx as nx
import sciris as sc
import datetime as dt

def fix_dates(g):
    for ax in g.axes.flat:
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter())
        #ax.xaxis.set_major_locator(mdates.MonthLocator())
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

def fix_yaxis(g):
    for ax in g.axes.flat:
        locator = mtick.MaxNLocator(nbins=5, min_n_ticks=5)
        #formatter = mdates.ConciseDateFormatter(locator)
        ax.yaxis.set_major_locator(locator)
        #ax.yaxis.set_major_formatter(formatter)

def fix_axis_labels(g, prefix=None):
    for i in range(g.axes.shape[0]):
        text = g.axes[i,-1].texts[0].get_text()
        if prefix is not None:
            text = prefix + text
        g.axes[i,0].set_ylabel(text)
        g.axes[i,-1].texts[0].set_text('')
    return

def plot_scenarios(df, figdir, channels=None, var1='cov', var2='channel', slice_year=-1):
    sns.set_theme(font_scale=1.2, style='whitegrid')

    # Renaming
    df.replace({'rng': {'centralized':'Centralized', 'multi': 'CRN'}}, inplace=True)
    rngs = ['Centralized', 'CRN']

    first_year = int(df['year'].iloc[0])
    assert df['year'].iloc[0] == first_year
    df['date'] = pd.to_datetime(365 * (df['year']-first_year), unit='D', origin=dt.datetime(year=first_year, month=1, day=1)) #pd.to_datetime(df['year'], format='%f', unit='Y')

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

    id_vars = ['date', 'rand_seed', cov, 'rng', 'network', 'eff', 'n_agents']
    d = pd.melt(df, id_vars=['date', 'rand_seed', cov, 'rng', 'network', 'eff', 'n_agents'], var_name='channel', value_name='Value')

    # Slice to the one channel specified by the user
    d = d.loc[d['channel'].isin(channels)]

    if var1 != 'channel' and var2 != 'channel':
        assert len(channels)==1, 'If not slicing over channel, only one channel can be provided.'

    d['baseline'] = d[cov]=='Reference'
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]

    id_vars.append('channel')
    bl = bl.set_index(id_vars)[['Value']].reset_index(cov)
    scn = scn.set_index(id_vars)[['Value']].reset_index(cov)

    id_vars.remove(cov)
    mrg = scn.merge(bl, on=id_vars, suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg = mrg.sort_index()
    mrg['Coverage'] = pd.Categorical(mrg['Coverage'], categories=cov_ord[:-1]) # Agh^2!

    id_vars = ['date', cov, 'rng', 'network', 'eff', 'n_agents', 'channel']
    cor = mrg.reset_index().groupby(id_vars, observed=True)[['Value', 'Value_ref']].apply(lambda x: np.corrcoef(x['Value'], x['Value_ref'], rowvar=False)[0,1])
    cor.name = 'Pearson'
    #cor.replace([np.inf, -np.inf, np.nan], 1, inplace=True)

    kw = {'height': 3, 'aspect': 1.4}
    fkw = {'sharey': 'row', 'sharex': 'col', 'margin_titles': True} # facet_kws={**fkw, **{'sharey':'row'}}

    var1_ord = None
    if var1=='cov':
        var1 = cov
        var1_ord = None # Categorical, so not needed

    var2_ord = None # Categorical, so not needed
    if var2=='cov':
        var2 = cov
        var2_ord = None # Categorical, so not needed

    rng_colors = [(0,0,0), (1,0,0)] # [(0,1,1), (1,0,1)]

    
    #%% TIMESERIES

    def plot_median(data, **kws):
        ax = plt.gca()
        sns.lineplot(data=data, x='date', y='Value', hue=var1, hue_order=var1_ord,
            palette='Set1', errorbar=('pi', 95), estimator=np.median, lw=2, ax=ax)
        return

    #g = sns.relplot(kind='line', data=d, x='date', y='Value', hue=var1, hue_order=var1_ord, row=var2, col='rng', col_order=rngs,
    #    palette='Set1', facet_kws=fkw, **kw, errorbar=('pi', 25), estimator=np.median, lw=2)
    g = sns.relplot(kind='line', data=d, x='date', y='Value', hue=var1, hue_order=var1_ord, row=var2, col='rng', col_order=rngs,
        palette='Set1', facet_kws=fkw, **kw, units='rand_seed', estimator=None, lw=0.05, alpha=0.5)
    g.map_dataframe(plot_median)

    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.set_xlabels('Date')
    fix_dates(g)
    fix_yaxis(g)
    fix_axis_labels(g)
    g.figure.savefig(os.path.join(figdir, 'timeseries.png'), bbox_inches='tight', dpi=300)
    plt.close(g.figure)


    #%% TIMESERIES: specific channels
    for ch in ['Maternal Deaths', 'Infected']:
        if ch in d['channel'].unique():
            g = sns.relplot(kind='line', data=d, x='date', y='Value', hue=var1, hue_order=var1_ord, row='channel', row_order=[ch], col='rng', col_order=rngs,
                palette='Set1', errorbar='sd', lw=2, facet_kws=fkw, **kw)
            g.set_titles(col_template='{col_name}', row_template='{row_name}')
            g.set_xlabels('Date')
            fix_dates(g)
            fix_yaxis(g)
            g.figure.savefig(os.path.join(figdir, f'timeseries_{ch.replace(" ", "")}.png'), bbox_inches='tight', dpi=300)
            plt.close(g.figure)


    #%% DIFF TIMESERIES
    for ms, mrg_by_ms in mrg.groupby('rng', observed=True):
        g = sns.relplot(kind='line', data=mrg_by_ms, x='date', y='Value - Reference', hue=var1, hue_order=var1_ord, col=var2, #col_order=var2_ord,
                palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw, **kw)
        g.set_titles(col_template='{col_name}', row_template='{row_name}')
        #g.figure.suptitle('MultiRNG' if ms else 'SingleRNG')
        g.figure.subplots_adjust(top=0.88)
        g.set_xlabels('Date')
        fix_dates(g)
        fix_yaxis(g)
        g.figure.savefig(os.path.join(figdir, f'diff_{ms}.png'), bbox_inches='tight', dpi=300)
        plt.close(g.figure)


    #%% SLICE AT SPECIFIC TIME
    if slice_year < 0:
        slice_year = d['date'].max()
    else:
        years = df['year'].unique()
        dates = d['date'].unique()
        sidx = sc.findnearest(years, slice_year)
        slice_year = dates[sidx]

    slice_str = slice_year.strftime('%b %-d, %Y')

    facet_kws = fkw.copy()
    facet_kws['sharey'] = 'row'
    facet_kws['sharex'] = 'row'

    mtf = mrg.loc[slice_year]
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue=var1, hue_order=var1_ord, x='Value - Reference',
            row=var2, col='rng', col_order=rngs, facet_kws=facet_kws, palette='Set1', **kw)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.set_xlabels(f'Value - Reference on {slice_str}')
    fix_yaxis(g)
    g.figure.savefig(os.path.join(figdir, 'slice.png'), bbox_inches='tight', dpi=300)
    plt.close(g.figure)

    facet_kws['sharey'] = False
    facet_kws['sharex'] = 'col'
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue='rng', hue_order=rngs, x='Value - Reference',
            col=var2, row=var1, facet_kws=facet_kws, palette='Set1', **kw)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.set_xlabels(f'Value - Reference on {slice_str}')
    fix_yaxis(g)
    g.figure.savefig(os.path.join(figdir, 'slice2.png'), bbox_inches='tight', dpi=300)
    plt.close(g.figure)


    #%% COR SCATTER AT SLICE TIME
    ctf = mtf.reset_index('rand_seed')
    if var1 in ctf.columns:
        ctf.set_index(var1, append=True)
    ctf = ctf.sort_index()
    #if len(channels) > 1:
    #    g = sns.relplot(data=ctf, kind='scatter', hue='rng', hue_order=rngs, style='rng', style_order=rngs, x='Value_ref', y='Value', size='rng', sizes={'Centralized':10, 'CRN': 10},
    #        col='channel', col_order=channels, row=var1, row_order=var1_ord, facet_kws=fkw, palette=rng_colors, **kw)
    #else:
    facet_kws = fkw.copy()
    facet_kws['sharey'] = 'row'
    facet_kws['sharex'] = 'row'

    g = sns.relplot(data=ctf, kind='scatter', hue='rng', hue_order=rngs, x='Value_ref', y='Value', size='rng', sizes={'Centralized':25, 'CRN': 25},
        style='rng', style_order=rngs, markers={'Centralized':'+', 'CRN': 'x'},
        col=var1, col_order=var1_ord, row=var2, facet_kws=facet_kws, palette=rng_colors, **kw, lw=None)

    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.set_xlabels(f'Reference on {slice_str}')
    g.set_ylabels(f'Value on {slice_str}')
    #fix_axis_labels(g, postfix=f'as of {slice_str}')
    fix_yaxis(g)
    g.figure.savefig(os.path.join(figdir, 'cor_slice.png'), bbox_inches='tight', dpi=300)
    plt.close(g.figure)

    # Share y from here on
    fkw['sharey'] = 'row'


    #%% DIFF TIMESERIES ALL IN ONE
    g = sns.relplot(kind='line', data=mrg, x='date', y='Value - Reference', hue=var1, hue_order=var1_ord, row=var2, #row_order=channels,
            col='rng', col_order=rngs, palette='Set1', estimator=None, units='rand_seed', lw=0.1, facet_kws=fkw, **kw)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.figure.subplots_adjust(top=0.88)
    g.set_xlabels('Date')
    fix_dates(g)
    fix_yaxis(g)
    fix_axis_labels(g, prefix='Difference in\n')
    g.figure.savefig(os.path.join(figdir, f'diff.png'), bbox_inches='tight', dpi=300)
    plt.close(g.figure)


    #%% CORRELATION OVER TIME
    try:
        if isinstance(cor, pd.Series):
            corf = cor.to_frame()
        else:
            corf = cor
        g = sns.relplot(kind='line', data=corf, x='date', y='Pearson', hue='rng', hue_order=rngs,
                col=var1, col_order=var1_ord, row=var2, row_order=var2_ord, style='rng', style_order=rngs,
                palette=rng_colors, errorbar='sd', lw=2, facet_kws=fkw, **kw)
        g.set_titles(col_template='{col_name}', row_template='{row_name}')
        g.figure.subplots_adjust(top=0.88)
        g.set_xlabels('Date')
        fix_dates(g)
        fix_yaxis(g)
        fix_axis_labels(g)
        g.figure.savefig(os.path.join(figdir, 'cor.png'), bbox_inches='tight', dpi=300)
        plt.close(g.figure)
    except Exception as e:
        print('CORRELATION OVER TIME did not work')
        print(e)

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
