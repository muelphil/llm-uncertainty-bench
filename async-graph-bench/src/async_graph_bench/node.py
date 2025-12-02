from typing import Protocol, List, Dict, Union, Optional, Any, Awaitable


class Node(Protocol):
    id: Optional[str]
    requires: Optional[List[str]]
    provides: List[str]
    description: Optional[str]
    spread: Optional[bool]

    def __call__(self, item_stats: Dict[str, List[Any]], **kwargs: Any) -> Union[
        Dict[str, List[Any]],
        Any,
        Awaitable[Union[Dict[str, List[Any]], Any]]
    ]:
        ...
