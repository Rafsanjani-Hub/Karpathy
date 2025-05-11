from graphviz import Digraph

def trace_routes(node):
    '''
    The trace_routes function determines nodes and edges.
    '''

    nodes = []
    edges = []

    def build_topology(node):
        if node not in nodes:
            nodes.append(node)
            for child in node._children:
                edges.append((child, node)) #u -> v
                build_topology(child)
            #end-for
        #end-if/else
    #end-def
    build_topology(node)

    return nodes, edges
#end-def


def connect_dots(node):
    nodes, edges = trace_routes(node)
    
    dots = Digraph(format='svg', graph_attr={'rankdir':'LR'})

    for node in nodes:
        uid = str(id(node))

        dots.node(name=uid, label=f"{node.label}|data = {node.data:.4f}|grad = {node.grad:.4f}", color='blue', shape='record')
        
        if node._operation != '':
            dots.node(name=uid + node._operation, label=node._operation, color='blue', shape='circle')
            dots.edge(uid + node._operation, uid, color='green')
        #end-if/else
    #end-for

    for node1, node2 in edges:
        dots.edge(str(id(node1)), str(id(node2)) + node2._operation, color='green')
    #end-for

    return dots
#end-def