import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random
import sys
from typing import Tuple, Set
from sklearn.model_selection import train_test_split
from copy import deepcopy
import math

class RecNN(nn.Module):

    def __init__(self):
        super(RecNN, self).__init__()
        # A network for each type of node
        self.length = 1
        self.terminal = nn.Linear(1, self.length)
        self.plus = nn.Linear(2 * self.length, self.length)
        self.minus = nn.Linear(2 * self.length, self.length)
        self.network_map = {
            "terminal": self.terminal,
            "+": self.plus,
            "-": self.minus
        }
        # A network for each
        self.final = nn.Identity(self.length, 1, bias=False)

    def forward(self, graph: nx.DiGraph):
        assert nx.is_directed_acyclic_graph(graph)
        # We sort the graph topologically - i.e. each node is processed before it's successors.
        # The sink is the last entry in this list.
        plan = list(nx.topological_sort(graph))
        # We mark the sink so we can - later - apply the final layer and return the result
        last = plan[-1]
        for node in plan:
            if isinstance(node.value, float):
                # if the input is a source, use a different network
                network = self.network_map["terminal"]
                output = network(torch.tensor([node.value]))
            else:
                # if it is an internal node, we have to determine which kind of network to use
                network = self.network_map[node.value]
                # collect the output from all predecessors and feed it to the network
                cat = torch.cat(node.inputs, 0)
                output = network(cat)
            if node == last:
                # The sink generates the final result
                return self.final(output)
            else:
                # Every non-sink-node passes its output to all its successors
                parents = list(graph.successors(node))
                for parent in parents:
                    parent.inputs.append(output)


class Node:
    def __init__(self, id, value):
        self.id = id
        self.value = value
        self.inputs = []


def execute_network(net, loss_fn, optimizer, train, test, valid, epochs=100):
    for epoch in range(epochs):
        running_loss = 0
        for x, y in train:
            # Reset gradients from previous datapoint
            optimizer.zero_grad()
            # Run the network
            prediction = net(deepcopy(x))
            # Calculate loss
            loss = loss_fn(prediction, y)
            # Calculate gradients
            loss.backward()
            # Change parameters
            optimizer.step()
            running_loss += loss.item()
    test_loss = sum(loss_fn(net(deepcopy(x)), y).item() for x, y in test) / len(test)
    print("test_loss:", test_loss)


# ----
# Everything below here is just data generation and execution
# ---

def solve(graph: nx.DiGraph) -> float:
    def _subsolve(node: Node):
        if isinstance(node.value, float):
            return node.value
        else:
            kids = list(graph.predecessors(node))
            assert len(kids) == 2, kids
            if node.value == "+":
                return -0.8 * (_subsolve(kids[0]) + _subsolve(kids[1]))
            elif node.value == "-":
                return _subsolve(kids[0]) + _subsolve(kids[1]) + 0.3
            else:
                raise ValueError(node.value)

    root = list(nx.topological_sort(graph))[-1]
    return _subsolve(root)


def graph_to_string(graph: nx.DiGraph):
    def tostr(node):
        if isinstance(node.value, float):
            return str(node.value)
        else:
            kids = list(graph.predecessors(node))
            if node.value == "+":
                return f"(-2 * ({tostr(kids[0])} + {tostr(kids[1])}))"
            elif node.value == "-":
                return f"({tostr(kids[0])} + {tostr(kids[1])} + 5)"
            else:
                raise Exception(node.value)
    return tostr(next(reversed(list(nx.topological_sort(graph)))))


def generate_tree(name, max_depth) -> Tuple[Node, Set[Tuple[Node, Node]]]:
    if random.random() < 0.3 or len(name) >= max_depth:
        root = Node(name, int((0.5 - random.random())*1000)/100)
        return root, set()
    else:
        if random.random() < 0.5:
            operator = "+"
        else:
            operator = "-"
        root = Node(name, operator)
        left, left_edges = generate_tree(name + "l", max_depth)
        right, right_edges = generate_tree(name + "r", max_depth)
        return root, left_edges.union(right_edges).union({(left, root), (right, root)})


def generate_DAG(max_depth):
    root, edges = generate_tree("", max_depth)
    if edges:
        return nx.DiGraph(edges)
    else:
        g = nx.DiGraph()
        g.add_node(root)
        return g


def main():
    net = RecNN()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    dataset = []
    for depth in range(0, 10):
        print("depth", depth)
        n = 100
        dataset += [(t, torch.tensor([solve(t)])) for _ in range(n) for t in [generate_DAG(depth)]]
        train, temp = train_test_split(dataset, train_size=0.7)
        test, valid = train_test_split(temp, train_size=0.7)
        execute_network(net, loss_fn, optimizer, train, test, valid)


if __name__ == "__main__":
    main()
