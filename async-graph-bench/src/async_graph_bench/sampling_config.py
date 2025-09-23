import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingConfig:
    sampling_size: int
    all_variations: Optional[bool] = None
