import logging
from collections import defaultdict, deque
from typing import Set, List, DefaultDict, Dict, Tuple, Union

from .data_source import DataSource
from .node_config import NodeConfig

log = logging.getLogger(__name__)


class AcyclicDirectedGraph:
    """
    Represents an acyclic directed graph, featuring a data_source and nodes for calculation of statistics (provided dependencies) based on
    required dependencies. The graph is built based on dependencies required and statistics provided by the nodes.
    """

    def __init__(self, data_source: DataSource, node_configs: List[NodeConfig]):
        """
        Initializes the graph with a data source and a list of calculator/consumer nodes.

        :param data_source: The single data source node with.
        :param node_configs: A list of all calculator and consumer nodes as NodeConfig instances.
        """
        self.data_source: DataSource = data_source
        self.optional_nodes_configs: Set[NodeConfig] = {node for node in node_configs if not node.greedy}
        self.greedy_nodes_configs: Set[NodeConfig] = {node for node in node_configs if node.greedy}
        # Adjacency list: producer → consumers
        self.edges: DefaultDict[Union[NodeConfig, DataSource], Set[NodeConfig]] = defaultdict(set)
        self.unreachable_nodes: Set[NodeConfig] = set()  # Nodes whose required dependencies cannot be resolved
        self.pruned_nodes: Set[NodeConfig] = set()  # Nodes removed by treeshaking
        self.sampling: Dict[NodeConfig, Set] = defaultdict(lambda: set())

    def get_all_predecessors(self, target_node):
        """
        Given a node, returns all nodes that have a path leading to it
        (excluding the node itself).
        """
        predecessors = set()

        def dfs(node):
            for producer, consumers in self.edges.items():
                if node in consumers and producer not in predecessors:
                    predecessors.add(producer)
                    dfs(producer)

        dfs(target_node)
        return predecessors

    def get_safe_edge_targets(self, from_node):
        """ Returns all target nodes that are safe targets for new edges from_node -> target_node without creating cyclic dependencies """
        forbidden = {from_node} | self.get_all_predecessors(from_node)
        all_nodes = {self.data_source} | self.optional_nodes_configs | self.greedy_nodes_configs
        return all_nodes - forbidden

    def get_all_successors(self, node):
        """
        Return all nodes reachable from `node` following outgoing edges.
        """
        visited = set()

        def dfs(n):
            for consumer in self.edges.get(n, ()):
                if consumer not in visited:
                    visited.add(consumer)
                    dfs(consumer)

        dfs(node)
        return visited

    def get_safe_edge_sources(self, target_node):
        """
        Return all nodes `source` such that adding edge `source -> target_node`
        will NOT create a cycle.
        """
        # Nodes that would create a cycle are the target itself and any node
        # reachable FROM the target (its successors).
        forbidden = self.get_all_successors(target_node) | {target_node}
        all_nodes = {self.data_source} | self.optional_nodes_configs | self.greedy_nodes_configs
        return all_nodes - forbidden

    def build_graph(self) -> None:
        """
        Builds the dependency graph taking sampling semantics into account.

        Implementation:
        - Nodes are added when all required (non-sampled) dependencies are satisfied by already
          added nodes.
        - If only sampled_... dependencies remain, a node may still be added but only under
          constraints that ensure "destructive" ('first') sampling does not break later nodes.
        - If a predecessor (or any of its predecessors) performs sampling in mode 'first',
          a node that depends on sampled_... keys may only be added if:
            * the node itself samples in mode 'first', and
            * there exists a predecessor that already samples the same sampling-config (or can
              be connected to providers of the missing base dependencies without creating cycles)
              so that that predecessor can produce the sampled_... keys in the same config.
        - Edges may be created from available provider nodes to predecessors to allow them to
          sample required dependencies, provided doing so does not create cycles.
        """
        nodes: List[Union[DataSource, NodeConfig]] = [self.data_source]  # already included/available producers
        not_visited = self.optional_nodes_configs | self.greedy_nodes_configs

        changed = True

        while changed:
            changed = False
            # iterate over a snapshot because we will mutate not_visited inside
            for consumer_node in list(not_visited):
                required_dependencies = set(consumer_node.requires)
                # potential previous nodes that already provide some required deps
                potential_previous_nodes: Set[NodeConfig] = set()

                # find direct providers among currently added nodes
                for producer_node in nodes:
                    provided = set(producer_node.provides).intersection(required_dependencies)
                    if provided:
                        required_dependencies -= provided
                        potential_previous_nodes.add(producer_node)
                        if not required_dependencies:
                            break

                # 1) If no required dependencies remain -> safe to add the consumer
                if not required_dependencies:
                    # add node and edges
                    for producer_node in potential_previous_nodes:
                        self.edges[producer_node].add(consumer_node)
                    nodes.append(consumer_node)
                    not_visited.remove(consumer_node)
                    changed = True
                    continue

                # 2) If some required_dependencies remain and some are NOT sampled_xxx -> cannot add now
                non_sampled_remaining = {d for d in required_dependencies if not d.startswith("sampled_")}
                if non_sampled_remaining:
                    # cannot add yet, wait for more providers
                    continue

                # 3) Only sampled_... dependencies remain
                # strip 'sampled_' prefix to get base deps required
                sampling_base_deps = {d[len("sampled_"):] for d in required_dependencies}
                missing_sampling_base_deps = {d[len("sampled_"):] for d in required_dependencies}

                # collect all predecessors (producers and their predecessors) to detect sampling upstream
                predecessors = set()
                for p in potential_previous_nodes:
                    predecessors |= {p} | self.get_all_predecessors(p)

                # find any predecessor that applies sampling (according to current sampling map)
                preds_with_sampling = [n for n in predecessors if not isinstance(n, DataSource) and n.is_sampling()]
                preds_with_first_mode = [n for n in preds_with_sampling if n.sampling_mode == "first"]

                added_deps = defaultdict(lambda: set())
                added_edges = list()

                # If no predecessor (transitively) performs 'first' sampling, the consumer may apply sampling itself.
                if not preds_with_first_mode:
                    for node in potential_previous_nodes:
                        predecessor_nodes = {node} | self.get_all_predecessors(node)
                        predecessor_stats = {s for n in predecessor_nodes for s in n.provides}
                        for base_dep in list(missing_sampling_base_deps):
                            if base_dep in predecessor_stats:
                                missing_sampling_base_deps.remove(base_dep)

                    if missing_sampling_base_deps:  # still not sufficient, checking if additional nodes can be used to get dep
                        for base_dep in list(missing_sampling_base_deps):
                            save_sources = self.get_safe_edge_sources(consumer_node)
                            for source in save_sources:
                                if base_dep in source.provides:
                                    potential_previous_nodes.add(source)
                                    added_deps[consumer_node].add(base_dep)
                                    missing_sampling_base_deps.remove(base_dep)

                    if not missing_sampling_base_deps:
                        for node, deps_to_add in added_deps.items():
                            self.sampling[node].update(deps_to_add)
                        self.sampling[consumer_node].update(sampling_base_deps)
                        for producer_node in potential_previous_nodes:
                            self.edges[producer_node].add(consumer_node)
                        nodes.append(consumer_node)
                        not_visited.remove(consumer_node)
                        changed = True
                        continue

                    # 4) There is at least one predecessor that uses 'first' sampling
                    # consumer_node must also sample in 'first' mode and be compatible with those predecessors
                    if consumer_node.sampling_mode != "first":
                        # consumer cannot be safely added: a predecessor is destructively sampling
                        # and the consumer does not also use first-mode sampling
                        continue

                    # REQUIREMENT: ALL preds_with_first_mode must share the same sampling config
                    if any(p.sampling_config != consumer_node.sampling_config for p in preds_with_first_mode):
                        # incompatible sampling configurations upstream -> cannot add consumer
                        continue

                    # Determine the set of base dependencies (without "sampled_" prefix) that the consumer needs

                    for node in preds_with_first_mode:
                        missing_sampling_base_deps -= set(self.sampling[node])  # these are already sampled

                    for node in preds_with_first_mode:
                        predecessor_nodes = self.get_all_predecessors(node)
                        predecessor_stats = {n.provides for n in predecessor_nodes}
                        for base_dep in list(missing_sampling_base_deps):
                            # if base_dep accessible by node add it
                            if base_dep in predecessor_stats:
                                # sampling[node]["deps"].add(base_dep)
                                added_deps[node].add(base_dep)
                                missing_sampling_base_deps.remove(base_dep)

                    if missing_sampling_base_deps:  # still not sufficient, checking if additional nodes can be used to get dep
                        for node in preds_with_first_mode:
                            for base_dep in list(missing_sampling_base_deps):
                                save_sources = self.get_safe_edge_sources(node)
                                for source in save_sources:
                                    if base_dep in source.provides:
                                        # self.edges[source].add(node)
                                        added_edges.append((source, node))
                                        # sampling[node]["deps"].add(base_dep)
                                        added_deps[node].add(base_dep)
                                        missing_sampling_base_deps.remove(base_dep)

                    if not missing_sampling_base_deps:
                        nodes.append(consumer_node)
                        for node in potential_previous_nodes:
                            self.edges[node].add(consumer_node)
                        for source, target in added_edges:
                            self.edges[source].add(target)
                        for node, deps_to_add in added_deps.items():
                            self.sampling[node].update(deps_to_add)

        # After fixed-point loop, any remaining not_visited nodes are unreachable.
        unreachable_greedy_nodes = not_visited.intersection(self.greedy_nodes_configs)
        if unreachable_greedy_nodes:
            for consumer in unreachable_greedy_nodes:
                # compute missing dependencies for logging purposes
                present_deps = set()
                for n in nodes:
                    present_deps.update(getattr(n, "provides", []))
                missing_dependencies = set(consumer.requires) - present_deps
                log.warning(
                    f'Removed unreachable consumer {consumer.id} due to missing required dependencies. '
                    f'(Missing Dependencies: {missing_dependencies}, Present Dependencies: {present_deps})'
                )
            # remove unreachable greedy nodes from the greedy set
            self.greedy_nodes_configs -= unreachable_greedy_nodes

        self.unreachable_nodes = set(not_visited)

    def log_unusable_nodes(self) -> None:
        """
        Logs all nodes that could not be included in the graph due to unresolved dependencies.
        """
        if not self.unreachable_nodes:
            log.info("All nodes are reachable.")
        else:
            log.warning("Unreachable nodes due to unmet required dependencies:")
            for node in self.unreachable_nodes:
                log.warning(f" - {node.id} (Required Dependencies: {node.requires})")

    def turn_consumer_non_greedy(self, consumer: NodeConfig) -> None:
        """
        Converts a greedy consumer into a non-greedy one.

        :param consumer: The consumer node to convert.
        """
        self.greedy_nodes_configs.remove(consumer)
        self.optional_nodes_configs.add(consumer)

    def treeshake(self) -> List[NodeConfig]:
        """
        Removes optional calculators that are not used by any downstream node.
        Prunes the graph until only useful optional nodes remain.

        :return: A list of pruned (removed) calculator nodes.
        """
        pruned_calculators = []
        while True:
            nodes_to_remove = [
                calculator
                for calculator in self.optional_nodes_configs
                if not self.edges[calculator]
            ]
            pruned_calculators += nodes_to_remove

            if not nodes_to_remove:
                break

            for calculator in nodes_to_remove:
                self.edges.pop(calculator, None)
                for parent in self.edges:
                    self.edges[parent].discard(calculator)
                self.optional_nodes_configs.remove(calculator)

        self.pruned_nodes.update(pruned_calculators)
        return pruned_calculators

    def _is_descendant_node(self, source_node: NodeConfig, node: NodeConfig) -> bool:
        """
        Checks if `node` is reachable from `source_node` (i.e. is a descendant in the graph).

        :param source_node: The node to start traversal from.
        :param node: The node to test reachability to.
        :return: True if `node` is a descendant of `source_node`, else False.
        """
        visited = set()
        queue = deque([source_node])

        while queue:
            current = queue.popleft()
            if current == node:
                return True
            if current not in visited:
                visited.add(current)
                queue.extend(self.edges[current])

        return False

    def remove_transient_edges(self) -> None:
        """
        Removes redundant edges for nodes with multiple parents. A redundant edge
        is one whose source is a descendant of another source, meaning it's unnecessary.
        """
        log.info("Checking for transient edges...")
        incoming_edges = defaultdict(list)
        for source, targets in self.edges.items():
            for target in targets:
                incoming_edges[target].append(source)

        for node, sources in incoming_edges.items():
            if len(sources) > 1:
                for source in sources:
                    for other_source in sources:
                        if source != other_source and self._is_descendant_node(other_source, source):
                            if node in self.edges[other_source]:
                                self.edges[other_source].remove(node)
                                log.info(f"Removed edge from {other_source.id} to {node.id}")

    def count_parents(self, node: NodeConfig) -> int:
        """
        Counts how many nodes have an edge leading to the given node.

        :param node: The node to count parents for.
        :return: The number of parent nodes.
        """
        return sum(node in children for children in self.edges.values())

    def track_consumers_by_calculator(self) -> DefaultDict[NodeConfig, List[NodeConfig]]:
        """
        Tracks all greedy consumers that are downstream of each optional calculator,
        by backtracking through the graph.

        :return: A mapping from calculator nodes to lists of their dependent consumers.
        """
        consumers_by_calculator: DefaultDict[NodeConfig, List[NodeConfig]] = defaultdict(list)

        for consumer in self.greedy_nodes_configs:
            queue = deque([consumer])
            visited = set()

            while queue:
                current_node = queue.popleft()

                if current_node in visited:
                    continue
                visited.add(current_node)

                for producer in self.edges:
                    if current_node in self.edges[producer]:
                        consumers_by_calculator[producer].append(consumer)
                        queue.append(producer)

        return consumers_by_calculator

    def is_leaf_node(self, node: NodeConfig) -> bool:
        """
        Checks if the given node is a leaf (i.e. has no children).

        :param node: The node to test.
        :return: True if the node has no outgoing edges, else False.
        """
        return len(self.edges[node]) == 0

    def copy(self) -> "AcyclicDirectedGraph":
        """
        Create a copy of this graph.
        Shallow copy lists and sets, deep copy dict values for sampling.
        """
        new_graph = object.__new__(AcyclicDirectedGraph)
        new_graph.data_source = self.data_source
        new_graph.optional_nodes_configs = set(self.optional_nodes_configs)
        new_graph.greedy_nodes_configs = set(self.greedy_nodes_configs)
        new_graph.edges = defaultdict(set, {k: set(v) for k, v in self.edges.items()})
        new_graph.unreachable_nodes = set(self.unreachable_nodes)
        new_graph.pruned_nodes = set(self.pruned_nodes)
        new_graph.sampling = defaultdict(set, {k: set(v) for k, v in self.sampling.items()})
        return new_graph

    def remove_node(self, node: NodeConfig) -> None:
        """
        Remove `node` and all its successors from the graph, updating edges and sets.
        """
        to_remove = self.get_all_successors(node) | {node}

        # Remove from edges
        for n in list(self.edges.keys()):
            if n in to_remove:
                del self.edges[n]
            else:
                self.edges[n] -= to_remove

        # Remove from optional/greedy sets
        self.optional_nodes_configs -= to_remove
        self.greedy_nodes_configs -= to_remove

        # Add removed nodes to pruned_nodes
        self.pruned_nodes |= to_remove

        # Remove from sampling
        for n in to_remove:
            self.sampling.pop(n, None)

    def get_nodes_and_edges_in_topological_order(self) -> Tuple[List[NodeConfig], List[Tuple[NodeConfig, NodeConfig]]]:
        """
        Traverse from data_source in breadth-first order and return:
          - all optional+greedy nodes in dependency-respecting order
          - all edges (from, to) in traversal order (parent edges before child edges).
        Will always return the nodes in the same order for the same graph, regardless of the order that the nodes were added.
        """
        all_nodes = self.optional_nodes_configs | self.greedy_nodes_configs
        seen, nodes, edges = set(), [], []
        queue: deque[Union[DataSource, NodeConfig]] = deque([self.data_source])

        while queue:
            node = queue.popleft()
            children = sorted(self.edges.get(node, []), key=lambda n: n.id)  # sorted for deterministic order
            for child in children:
                edges.append((node, child))
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
                    if child in all_nodes:
                        nodes.append(child)

        return nodes, edges
