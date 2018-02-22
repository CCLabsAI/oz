import oz

g = oz.KuhnPoker()

def print_indent(n, s):
    print("\t"*n + s)

nodes = {}

n_nodes = 0

def gen_node():
    global n_nodes
    name = "state{}".format(n_nodes)
    n_nodes = n_nodes + 1
    return name

def dot_game(g, node=gen_node(), h=[]):
    global nodes
    node_meta = nodes.setdefault(node, {})
    node_meta["history"] = h
    node_meta["player"] = g.player
    if g.is_terminal():
        nodes.setdefault(node, {})["reward"] = g.reward()
    else:
        for a in g.legal_actions():
            node_a = gen_node()
            label = str(a).split(".")[-1]
            l = '{} -> {} [label="{}"]'.format(node, node_a, label)
            print_indent(1, l)
            g_a = oz.KuhnPoker(g)
            g_a.act(a)
            dot_game(g_a, node_a, h + [a])

print("digraph {")

dot_game(g)


for node, meta in nodes.items():
    if 'reward' in meta:
        print_indent(1, '{} [label="{}" shape=plaintext]'.format(node, meta["reward"]))
    else:
        if meta['player'] == oz.KuhnPoker.Player.P1:
            shape_str = 'triangle'
        elif meta['player'] == oz.KuhnPoker.Player.P2:
            shape_str = 'invtriangle'
        elif meta['player'] == oz.KuhnPoker.Player.Chance:
            shape_str = 'circle'

        print_indent(1, '{} [label="" shape={}]'.format(node, shape_str))

print("}")
