import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random
import sys
from typing import Tuple, Set
from sklearn.model_selection import train_test_split

class RecNN(nn.Module):

    def __init__(self):
        super(RecNN, self).__init__()
        self.network_map = {
            "terminal": nn.Linear(1, 20),
            "+": nn.Linear(40, 20),
            "*": nn.Linear(40, 20)
        }
        self.final = nn.Linear(20, 1)

    def forward(self, graph: nx.DiGraph):
        assert nx.is_directed_acyclic_graph(graph)
        plan = list(nx.topological_sort(graph))
        last = plan[-1]
        for node in plan:
            if isinstance(node.value, float):
                network = self.network_map["terminal"]
                output = network(torch.tensor([node.value]))
            else:
                network = self.network_map[node.value]
                cat = torch.cat(node.inputs)
                output = network(cat)
            if node == last:
                return self.final(output)
            else:
                parents = list(graph.successors(node))
                for parent in parents:
                    parent.inputs.append(output)


class Node:
    def __init__(self, id, value):
        self.id = id
        self.value = value
        self.inputs = []


def solve(graph: nx.DiGraph) -> float:
    def _subsolve(node: Node):
        if isinstance(node.value, float):
            return node.value
        else:
            kids = list(graph.predecessors(node))
            assert len(kids) == 2, kids
            if node.value == "+":
                return _subsolve(kids[0]) + _subsolve(kids[1])
            elif node.value == "*":
                return _subsolve(kids[0]) * _subsolve(kids[1])
    root = list(nx.topological_sort(graph))[-1]
    return _subsolve(root)


def generate_tree(name) -> Tuple[Node, Set[Tuple[Node, Node]]]:
    if random.random() < 0.5 or len(name) > 10:
        root = Node(name, (0.5 - random.random()) * 10)
        return root, set()
    else:
        if random.random() < 0.9:
            operator = "+"
        else:
            operator = "*"
        root = Node(name, operator)
        left, left_edges = generate_tree(name + "l")
        right, right_edges = generate_tree(name + "r")
        return root, left_edges.union(right_edges).union({(left, root), (right, root)})


def generate_DAG():
    root, edges = generate_tree("")
    if edges:
        return nx.DiGraph(edges)
    else:
        g = nx.DiGraph()
        g.add_node(root)
        return g


if __name__ == "__main__":
    n = 10
    epochs = 2
    dataset = [(t, torch.tensor([solve(t)])) for _ in range(n) for t in [generate_DAG()]]
    train, test = train_test_split(dataset, test_size=0.9)
    net = RecNN()
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for x, y in train:
            prediction = net(x)
            loss = loss_fn(prediction, y)
            loss.backward()
            nx.set_node_attributes(x, )
            print(loss)
