# Analyzer used to plot a graphical representation of contact networks over time

import starsim as ss
import numpy as np
import pandas as pd
import networkx as nx

network = 'EmbeddingNet'

class Graph():
    def __init__(self, nodes, edges):
        self.graph = nx.from_pandas_edgelist(edges, source='p1', target='p2', edge_attr=True)
        self.graph.add_nodes_from(nodes.index)
        nx.set_node_attributes(self.graph, nodes.transpose().to_dict())
        return

    def draw_nodes(self, filter_fn, pos, ax, **kwargs):
        inds = [i for i, n in self.graph.nodes(data=True) if filter_fn(n)]
        nc = ['red' if nd['hiv'] else 'lightgray' for i, nd in self.graph.nodes(data=True) if i in inds]
        ec = ['green' if nd['on_art'] else 'black' for i, nd in self.graph.nodes(data=True) if i in inds]
        if inds:
            nx.draw_networkx_nodes(self.graph, nodelist=inds, pos=pos, ax=ax, node_color=nc, edgecolors=ec, **kwargs)
        return

    def plot(self, pos, edge_labels=False, ax=None):
        kwargs = dict(node_shape='x', node_size=250, linewidths=2, ax=ax, pos=pos)
        self.draw_nodes(lambda n: n['dead'], **kwargs)

        kwargs['node_shape'] = 'o'
        self.draw_nodes(lambda n: not n['dead'] and n['female'], **kwargs)

        kwargs['node_shape'] = 's'
        self.draw_nodes(lambda n: not n['dead'] and not n['female'], **kwargs)

        nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
        nx.draw_networkx_labels(self.graph, labels={i: int(a['uid']) for i, a in self.graph.nodes(data=True)}, font_size=8, pos=pos, ax=ax)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, edge_labels={(i, j): int(a['dur']) for i, j, a in self.graph.edges(data=True)}, font_size=8, pos=pos, ax=ax)
        return


class GraphAnalyzer(ss.Analyzer):
    ''' Simple analyzer to assess if random streams are working '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize the Analyzer object

        self.graphs = {}
        return

    def init_pre(self):
        self.initialized = True
        self.apply(self.sim, init=True)
        return

    def apply(self, sim, init=False):
        ever_alive = np.isfinite(sim.people.age.values)
        nodes = pd.DataFrame({
            'uid': sim.people.uid.values[ever_alive],
            'age': sim.people.age.values[ever_alive],
            'female': sim.people.female.values[ever_alive],
            'dead': ~sim.people.alive.values[ever_alive],
            'hiv': sim.people.hiv.infected.values[ever_alive],
            'on_art': sim.people.hiv.on_art.values[ever_alive],
        })

        edges = sim.networks[network.lower()].to_df()

        idx = sim.t.ti if not init else -1
        self.graphs[idx] = Graph(nodes, edges)
        return

    def finalize(self):
        super().finalize()
        return
