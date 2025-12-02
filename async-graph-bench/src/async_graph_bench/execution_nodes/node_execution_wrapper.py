import asyncio
import logging
from typing import Any, Dict, List, Union, AsyncIterator

from ..node_config import NodeConfig
from ..utils.end_of_data import EndOfData

log = logging.getLogger(__name__)


class NodeExecutionWrapper:
    """
    Wraps a processing node and handles its execution on batches of items.

    This class is responsible for:
    - Collecting inputs for the node from the given items.
    - Handling both async and sync node calls.
    - Merging the results back into the original items if the node is an intermediate step.
    - Formatting the output if the node is a consumer node (final output).

    Args:
        node: The node instance to execute. Must have a callable interface and a
              `requires` property, and optionally a `provides` property.
    """

    def __init__(self, node):
        self.node = node

    async def run_node(
            self,
            items: List[Dict[str, Any]],
            *args,
            **kwargs
    ) -> Union[Dict[str, List[Any]], List[Any]]:
        """
        Aggregates the input items based on node dependencies and runs the node on the inputs.

        Returns:
            Either a dict of lists (for intermediate nodes) or a list of final results.
        """
        combined_items = dict()
        for key in self.node.requires:
            combined_items[key] = []

        for item in items:
            for key in self.node.requires:
                combined_items[key].append(item[key])

        if asyncio.iscoroutinefunction(self.node.__call__):
            return await self.node(combined_items, *args, **kwargs)
        else:
            return self.node(combined_items, *args, **kwargs)

    async def execute(
            self,
            items: Union[Dict[str, Any], List[Dict[str, Any]], EndOfData],
            *args,
            **kwargs
    ) -> AsyncIterator[Union[Dict[str, Any], EndOfData]]:
        """
        Executes the wrapped node and yields processed items.

        Handles EndOfData signaling and distinguishes between intermediate and
        consumer nodes based on the presence of a `provides` attribute.
        """
        if isinstance(items, EndOfData):
            log.debug(f"{self.node.__class__.__name__} received end of data signal...")
            yield items
            return

        if not isinstance(items, list):
            items = [items]

        results = await self.run_node(items, *args, **kwargs)

        if hasattr(self.node, 'provides'):  # intermediate node
            for key, value in results.items():
                assert len(value) == len(items), f"Length mismatch! len(value)={len(value)}, len(items)={len(items)}"

                for i in range(len(value)):
                    items[i][key] = value[i]
            for item in items:
                yield item
        else:  # consumer node (final result)
            if isinstance(results, dict) and all(
                    isinstance(v, (list, tuple)) and len(v) == len(items) for v in results.values()):
                for idx, item in enumerate(items):
                    to_serialize = dict()
                    to_serialize["id"] = item["id"]
                    if "iter" in item:
                        to_serialize["iter"] = item["iter"]
                    for key, value in results.items():
                        to_serialize[key] = value[idx]
                    yield to_serialize
            else:
                result_prop_identifier = (NodeConfig.base_config or {}).get("prop_name", "value")
                for i in range(len(results)):
                    output = {
                        "id": items[i]["id"],
                        result_prop_identifier: results[i]
                    }
                    if "iter" in items[i]:
                        output["iter"] = items[i]["iter"]
                    yield output
