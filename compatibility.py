import itertools
from typing import Collection, List, Sequence, Set
import numpy as np


def tree_path_partition(mu_repr: Collection[np.ndarray], sigma: Sequence[int]):
    """
    Gives a tree-path partition of a set of vectors, all with the same length.
    :param mu_repr: A tree-child compatible set.
    :param sigma: A permutation of {1, ..., n} such that mu_rho[sigma[i]] <= mu_rho[sigma[j]] for all 1 <= i < j <= n,
                  where mu_rho is the maximum element in M.
    :return: The tree-path partition of M.
    """
    partition = [[] for _ in sigma]
    for mu in mu_repr:
        for i in sigma:
            if mu[i]:
                partition[i].append(mu)
                break
    return partition


def maximal_tree_paths(mu_repr: Collection[np.ndarray]) -> List[List[np.ndarray]]:
    """
    Recovers the maximal tree-path cover of a tree-child compatible set.
    :param mu_repr: A tree-child compatible set.
    :return: The maximal tree-path cover of the input set.
    """
    mu_rho = max(mu_repr, key=tuple)  # node associated to the root
    tree_cover: List[List[np.ndarray]] = [[] for _ in mu_rho]

    # get a permutation of {1, ..., n} such that
    # we can visit each label in a non-decreasing order of mu_rho[i]
    _, sigma = zip(*sorted(zip(mu_rho, itertools.count())))

    for mu in mu_repr:
        first = -1
        for i in sigma:  # visit each label in {1, ..., n} in the order defined by sigma
            if mu[i] > 0:
                if first == -1:
                    first = mu_rho[i]
                elif first < mu[i]:
                    break
                tree_cover[i].append(mu)
    return tree_cover


def reconstruction_network(mu_repr: Collection[np.ndarray]) -> List[Set[int]]:
    """
    Checks for tree-path compatibility and reconstructs the corresponding tree-child network with a time complexity of
    O(nm(n + log m)).
    :param mu_repr: A set of unsigned integer vectors, all of the same length.
    :return: The tree-child network associated to the inpu mu-representation, in an adjacency list representation, in
             case that the input set is tree-child compatible.
    :raise AssertionError: If the input mu-representation is not tree-child compatible.
    """
    mu_rho = max(mu_repr, key=tuple)
    _, sigma = zip(*sorted(zip(mu_rho, itertools.count())))
    p = tree_path_partition(mu_repr, sigma)
    adjacency_list: List[Set[int]] = []
    maximum_nodes: List[int] = [-1] * len(sigma)  # list containing the indices of the maximum nodes for each tree-path in p
    for i in range(len(sigma) - 1, -1, -1):
        sigma_i = sigma[i]
        p[sigma_i].sort(key=tuple)  # sort the elements of p lexicographically in a non-decreasing order

        # delta_i = (0,0,...,0,1,0,...,0)
        delta = np.zeros_like(mu_rho)
        delta[sigma_i] = 1
        adjacency_list.append(set())  # add a leaf node with no outcomming edges

        # assert that the minimum node is the node delta_i associated to the leaf
        assert np.all(p[sigma_i][0] == delta)

        # ensure that all nodes are less than the root up to the cartesian product order,
        # in particular the top node of the tree-path, since all of the other nodes are descendants of the maximal node.
        assert np.all(p[sigma_i][-1] <= mu_rho)

        for mu_prev, mu_next in itertools.pairwise(p[sigma_i]):
            adjacency_list.append(set())  # add a new node on top of p_i
            adjacency_list[-1].add(len(adjacency_list) - 2)  # add the previous top node of p_i as a descendant of the new node
            mu = np.round(mu_next - mu_prev)
            for k in range(i + 1, len(sigma)):
                if np.all(mu >= p[sigma[k]][-1]):
                    adjacency_list[-1].add(maximum_nodes[sigma[k]])
                    mu -= p[sigma[k]][-1]
            assert np.all(mu == 0)  # ensure that mu = 0

        maximum_nodes[sigma_i] = len(adjacency_list) - 1  # register the current top node as the maximum node of p_i

    return adjacency_list  # return the resulting tree-child network in form of an adjacency list
