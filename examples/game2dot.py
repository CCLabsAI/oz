import sys
from copy import copy

import oz

# g = oz.RockPaperScissors()
g = oz.KuhnPoker()

def print_indent(n, s):
    print("\t"*n + s)

n_nodes = 0

def gen_node():
    global n_nodes
    name = "state{}".format(n_nodes)
    n_nodes = n_nodes + 1
    return name

node_infoset = {}

def dot_game(g, node=gen_node(), h=[]):
    if (not g.is_terminal()) and (g.player != g.Player.Chance):
        infoset = g.infoset(g.player)
        ls = node_infoset.setdefault(infoset, [])
        ls.append(node)

    if g.is_terminal():
        l = '{} [label="{}" shape=plaintext]'.format(node, g.reward())
        print_indent(1, l)
    else:
        player = g.player
        if player == g.Player.P1:
            shape_str = 'triangle'
        elif player == g.Player.P2:
            shape_str = 'invtriangle'
        elif player == g.Player.Chance:
            shape_str = 'circle'

        props = 'label="" shape={}'.format(shape_str)
        print_indent(1, '{} [{}]'.format(node, props))

        for a in g.legal_actions():
            node_a = gen_node()
            g_a = copy(g)
            g_a.act(a)
            label = str(a).split(".")[-1]
            props = 'headport=_ tailport=c label="{}"'.format(label)
            if not g_a.is_terminal():
                props += ' weight=2'
            else:
                props += ' weight=1'
            l = '{} -> {} [{}]'.format(node, node_a, props)
            print_indent(1, l)
            dot_game(g_a, node_a, h + [a])

print('digraph {')

print_indent(1, 'graph [nodesep=0.2 ranksep=1.5 splines=true]')

print_indent(1, 'node [height=.5 width=.5]')
dot_game(g)

for infoset, nodes in node_infoset.items():
    print_indent(1, 'subgraph "cluster_infoset_{}" {{'.format(infoset))
    print_indent(2, 'label="{}"'.format(infoset))
    print_indent(2, 'style=dashed')
    for node in nodes:
        print_indent(2, node)
    print_indent(1, '}')

print('}')
