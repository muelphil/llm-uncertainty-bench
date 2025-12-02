import logging
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class ExceptionInfo:
    exception: Exception
    originator: Optional[Any] = None
    step: Optional[int] = None
