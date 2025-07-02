from typing import Iterator

import numpy as np
import random


def generate_binary(n: int, m: int) -> Iterator[np.ndarray]:
    assert n > 1 or m == 1, f"Number n of leafs must be at least 2 (found {n=})"
    assert m >= 2 * n - 1, f"Number m of nodes must be at least 2n-1 (found {n=}, {m=})"
    base = np.identity(n, int)
    p = [[np.copy(base[i])] for i in range(n)]
    sigma = list(range(n))
    random.shuffle(sigma)
    edges = [(sigma[random.randint(0, i - 1)], sigma[i]) for i in range(1, n)]
    for _ in range(m - 2 * n + 1):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        edges.append((sigma[i], sigma[j]))
    random.shuffle(edges)
    for i, j in edges:
        base[i][j] += 1
        p[i].append(np.copy(base[i]))
    base_inv = np.identity(n, int)
    for i in reversed(sigma):
        base_inv[i] = base[i] @ base_inv
    for i, p_i in enumerate(p):
        for mu in p_i:
            mu[i] -= 1
            mu = mu @ base_inv
            mu[i] += 1
            yield mu


if __name__ == '__main__':
    # randomly generate a tree-child compatible set and display the adjacency list of the associated tree-child network
    import compatibility
    mu_repr = list(generate_binary(3, 9))
    print("Mu-representation of the network:", *mu_repr)
    print("Adjacency list of the network:", compatibility.reconstruction_network(mu_repr))
