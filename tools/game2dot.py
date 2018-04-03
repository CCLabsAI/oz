import sys
from copy import copy

import oz

# import oz.game.flipguess
# import oz.game.kuhn
# import oz.game.leduk

# g = oz.game.flipguess.FlipGuess()
# g = oz.game.kuhn.KuhnPoker()
# g = oz.game.leduk.LedukPoker()

g = oz.make_kuhn()

infoset_style = 'cluster'
# infoset_style = 'edge'

infoset_nodes = {}
n_nodes = 0


def print_indent(n, s):
    print("\t"*n + str(s))


def gen_node():
    global n_nodes
    name = "state{}".format(n_nodes)
    n_nodes = n_nodes + 1
    return name


def print_dot(g, d, node=gen_node(), h=[]):
    if d <= 0:
        return

    if not g.is_terminal() and g.player != g.Player.Chance:
        infoset = g.infoset()
        ls = infoset_nodes.setdefault(infoset, [])
        ls.append(node)

    if g.is_terminal():
        r = g.utility(g.Player.P1)
        l = '{} [label="{}" shape=plaintext]'.format(node, r)
        print_indent(1, l)
    else:
        player = g.player
        if player == g.Player.P1:
            shape_str = 'triangle'
        elif player == g.Player.P2:
            shape_str = 'invtriangle'
        elif player == g.Player.Chance:
            shape_str = 'circle'
        else:
            raise ValueError('"{}" is not a valid player'.format(player))

        props = 'label="" shape={}'.format(shape_str)
        print_indent(1, '{} [{}]'.format(node, props))

        for a in g.infoset().actions:
            node_a = gen_node()
            g_a = copy(g)
            g_a.act(a)
            # label = a.name
            label = "action" # FIXME
            props = 'headport=_ tailport=c label="{}" len=1.5'.format(label)
            if not g_a.is_terminal():
                props += ' weight=4'
            else:
                props += ' weight=2'
            l = '{} -> {} [{}]'.format(node, node_a, props)
            print_indent(1, l)
            print_dot(g_a, d-1, node_a, h + [a])


def print_infoset_clusters(infoset_nodes):
    for infoset, nodes in infoset_nodes.items():
        print_indent(1, 'subgraph "cluster_infoset_{}" {{'.format(infoset))
        print_indent(2, 'label="{}"'.format(infoset))
        print_indent(2, 'style=dashed')
        for node in nodes:
            print_indent(2, node)
        print_indent(1, '}')


def print_infoset_edges(infoset_nodes):
    for infoset, nodes in infoset_nodes.items():
        for a, b in zip(nodes, nodes[1:]):
            props = 'dir=none style=dashed label="{}" weight=1 len=1'.format(infoset)
            print_indent(1, '{} -> {} [{}]'.format(a, b, props))


print_indent(0, 'digraph {')

print_indent(1, 'graph [nodesep=0.2 ranksep=1.5 splines=true]')
print_indent(1, 'node [height=.5 width=.5]')

print_dot(g, d=25)

if infoset_style == 'cluster':
    print_infoset_clusters(infoset_nodes)
elif infoset_style == 'edge':
    print_infoset_edges(infoset_nodes)

print_indent(0, '}')
