import time
from typing import Dict

import numpy as np


# from async_graph_bench.models.multi_nli_instances import RemoteEncoderModel

# bevor batching across generations: 1 min für 10.000 entries
class GreedyAlternativesSimpleNLICalculator:
    stats = ["assistant_tokens_alternatives_nli"]
    dependencies = ["assistant_tokens_decoded_alternatives"]

    async def __call__(self, dependencies: Dict[str, np.array], nli_model) -> Dict[
        str, np.ndarray]:  # nli_model : RemoteEncoderModel
        greedy_alternatives = dependencies["assistant_tokens_decoded_alternatives"]

        # start_time = time.time_ns()
        # total = (sum(len(t) for t in greedy_alternatives) * 4)

        # Send entire structure to worker
        greedy_alternatives_nli = await nli_model.encode(
            greedy_alternatives, batch_size=256
        )

        # end_time = time.time_ns()
        # elapsed_ms = (end_time - start_time) // 1_000_000
        # print(f"GreedyAlternativesSimpleNLICalculator worker call took {elapsed_ms}ms "
        #       f"for {total} token comparisons")

        return {"assistant_tokens_alternatives_nli": greedy_alternatives_nli}
