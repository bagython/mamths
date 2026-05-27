import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def generate_random_web_graph(n_nodes: int) -> nx.DiGraph:
    """Generates a random directed graph ensuring no dead ends (out-degree >= 1).

    Each node has between 1 and 3 outgoing edges. No self-loops are allowed.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        possible_targets = [node for node in range(n_nodes) if node != i]

        # Robustness safeguard: prevent ValueError if n_nodes < 4
        n_out = min(random.randint(1, 3), len(possible_targets))

        if n_out > 0:
            targets = random.sample(possible_targets, n_out)
            for t in targets:
                G.add_edge(i, t)

    return G


def build_pagerank_matrix(G: nx.DiGraph, alpha: float = 0.85) -> np.ndarray:
    """Constructs the positive, column-stochastic PageRank matrix B.

    B = alpha * A + (1 - alpha) * M
    where A is the normalized link matrix and M is the uniform transition
    matrix.
    """
    n_nodes: int = G.number_of_nodes()

    # adj_matrix[i, j] is 1 if link j -> i (taking transpose of networkx standard)
    adj_matrix: np.ndarray = nx.to_numpy_array(G).T

    # Normalize columns to form column-stochastic matrix A
    out_degrees: np.ndarray = adj_matrix.sum(axis=0)
    A: np.ndarray = adj_matrix / out_degrees

    # Create uniform teleportation matrix M
    M: np.ndarray = np.ones((n_nodes, n_nodes)) / n_nodes

    # Compute final transition matrix B
    B: np.ndarray = alpha * A + (1 - alpha) * M
    return B


def compute_pagerank_power_method(
    B: np.ndarray, tol: float = 1e-8, max_iter: int = 100
) -> np.ndarray:
    """Computes the stationary distribution (PageRank scores) using the iterative

    Power Method, scaling much better than exact eigenvalue decomposition.
    """
    n_nodes: int = B.shape[0]

    # Initialize with a uniform distribution vector
    x: np.ndarray = np.ones(n_nodes) / n_nodes

    for _ in range(max_iter):
        x_next: np.ndarray = B @ x

        # Check for convergence using L1 norm
        if np.linalg.norm(x_next - x, ord=1) < tol:
            return x_next
        x = x_next

    return x


def plot_pagerank_graph(G: nx.DiGraph, scores: np.ndarray) -> None:
    """Plots the web graph where node sizes are proportional to their PageRank.

    Includes a baseline minimum size so lower-ranked nodes remain visible.
    """
    plt.figure(figsize=(10, 7))

    # Base size of 300 to ensure visibility, scaling up by 8000 based on rank
    node_sizes: list[float] = [300.0 + (float(scores[i]) * 8000.0) for i in G.nodes()]

    # Layout for clean node distribution
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="skyblue", edgecolors="black"
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=15,
        edge_color="gray",
        alpha=0.7,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=11, font_color="black", font_weight="bold"
    )

    plt.title("PageRank Visualization (Node Size Proportional to Rank)", fontsize=14)
    plt.axis("off")
    plt.show()


def main():
    n_pages = 12
    web_graph = generate_random_web_graph(n_pages)

    pagerank_matrix = build_pagerank_matrix(web_graph, alpha=0.85)

    ranks = compute_pagerank_power_method(pagerank_matrix)

    print("Computed Ranks:")
    for node, rank in enumerate(ranks):
        print(f"Page {node}: {rank:.4f}")

    # 4) Plot the graph with proportional node sizes
    plot_pagerank_graph(web_graph, ranks)


if __name__ == "__main__":
    main()
