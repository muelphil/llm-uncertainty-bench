from typing import Protocol, List, Dict, Union, Optional, Any, Awaitable, TypeVar

T = TypeVar("T")  # generic type for data (could be np.ndarray, list, tensor, etc.)


class Node(Protocol[T]):
    id: Optional[str]
    dependencies: Optional[List[str]]
    stats: List[str]
    description: Optional[str]
    spread: Optional[bool]

    def __call__(self, item_stats: Dict[str, T], **kwargs: Any) -> Union[
        Dict[str, T],
        T,
        Awaitable[Union[Dict[str, T], T]]
    ]:
        ...
