import json
import logging
import random
import re
from collections import Counter
from collections import defaultdict
from typing import Dict
from typing import List, Optional

import jsonschema
import numpy as np
from async_graph_bench import GenerationParameters, Model

log = logging.getLogger(__name__)


def get_json_schema(n):
    return {
        "type": "object",
        "properties": {
            "simplified_values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "value": {"type": "string"}
                    },
                    "required": ["index", "value"]
                },
                "minItems": n,
                "maxItems": n
            },
            "cluster_ids": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "cluster_id": {"type": "string"}
                    },
                    "required": ["index", "cluster_id"]
                },
                "minItems": n,
                "maxItems": n
            }
        },
        "required": ["simplified_values", "cluster_ids"]
    }


# Prompt templates
clustering_prompt = """You are given a list of computed results, each with an **index** and a **value**. The value may contain measurement or rounding errors, so small deviations in decimal places are acceptable.
Your task is to **simplify** these values (if needed) and **cluster** them so that values differing only by small measurement/rounding errors are grouped together.

Follow this workflow **strictly**:

1. **Simplify and compute values**

   * Return each given value in a simplified and computed form.
   * If the given values state a unit and no unit is given for individual units, assume the unit from context
   * Simplify fractions, mathematical constants (e.g., \\pi), or decimal/scientific multipliers (e.g., `10^7`).
   * Preserve valid numeric units when present.

2. **Cluster the values**

   * Group numeric values into clusters if they represent approximately the same value, but allowing for small measurement/rounding differences such as 50.7 atm instead of 50.683 atm. Take units into consideration, for example: 65.8 J/mol does not equal 65.8 kJ/mol.
   * Assign each cluster an integer `cluster_id`, starting from `0`.
   * For non-numeric or invalid values (e.g., `<NONE>` or textual responses), assign `"error"` as the `cluster_id`.
   * Negative numbers, fractions, and zero values **with valid units** count as numeric results (not errors).

Return the final result as a **JSON object** with the following properties:

* `simplified_values`: a list of objects containing each input `index` and its simplified `value` with unit if present.
* `cluster_ids`: a list of objects containing each input `index` and its assigned `cluster_id`.

### Example

**Values (given in kJ/mol):**

[
    {{"index": 0, "value": "65.8 kJ/mol"}},
    {{"index": 1, "value": "I could not reach any conclusion"}},
    {{"index": 2, "value": "0.658 x 10^2 J/mol"}},
    {{"index": 3, "value": "65.8 J/mol"}},
    {{"index": 4, "value": "6549/100 kJ/mol"}},
    {{"index": 5, "value": "65.8 kJ/mol"}},
    {{"index": 6, "value": "\\frac{{131.6}}{{2}} kJ/mol"}},
    {{"index": 7, "value": "<NONE>"}},
    {{"index": 8, "value": "65.8 kJ/mol"}},
    {{"index": 9, "value": "64 kJ/mol"}}
]

**Result:**
{{
    "simplified_values": [
        {{"index": 0, "value": "65.8 kJ/mol"}},
        {{"index": 1, "value": "I could not reach any conclusion"}},
        {{"index": 2, "value": "0.0658 kJ/mol"}},
        {{"index": 3, "value": "0.0658 kJ/mol"}},
        {{"index": 4, "value": "65.49 kJ/mol"}},
        {{"index": 5, "value": "65.8 kJ/mol"}},
        {{"index": 6, "value": "65.8 kJ/mol"}},
        {{"index": 7, "value": "<NONE>"}},
        {{"index": 8, "value": "65.8 kJ/mol"}},
        {{"index": 9, "value": "64 kJ/mol"}}
    ],
    "cluster_ids": [
        {{"index": 0, "cluster_id": "0"}},
        {{"index": 1, "cluster_id": "error"}},
        {{"index": 2, "cluster_id": "1"}},
        {{"index": 3, "cluster_id": "1"}},
        {{"index": 4, "cluster_id": "2"}},
        {{"index": 5, "cluster_id": "1"}},
        {{"index": 6, "cluster_id": "1"}},
        {{"index": 7, "cluster_id": "error"}},
        {{"index": 8, "cluster_id": "1"}},
        {{"index": 9, "cluster_id": "3"}}
    ]
}}

Use the same workflow and output structure for the following input:

**Values{unit_clause}:**

{values}

**Result:**
"""

correct_cluster_prompt = """You are given several **clusters of numeric values**.
Each cluster groups values that are approximately equal, allowing for small deviations due to rounding or measurement errors.

Your task is to **identify the `cluster_id`** that matches the given correct answer:

* If the correct answer matches the numeric values of a cluster (within minor rounding/measurement differences), return the `cluster_id` of that cluster.
* If the correct answer does **not** match any cluster, return **NONE**.

### Important Instructions

* Follow this workflow **strictly**.
* Provide **only** the `cluster_id` (or `NONE`) as the output.

Clusters (each cluster is introduced as a bullet list under a header of the form `Cluster with cluster_id {{cluster_id}}:`):

{clusters}

Correct answer:

{correct_answer}{unit}

`cluster_id` associated with the correct answer (or NONE):
"""


def format_clustering_prompt(values: List[str], unit: str) -> str:
    unit_clause = f" (given in {unit})" if unit and unit.strip() else " (values without unit)"
    values_as_dict = [{"index": i, "value": item.replace('<|eot_id|>', '')} for i, item in enumerate(values)]
    values_as_json = json.dumps(values_as_dict, indent=2)
    return clustering_prompt.format(unit_clause=unit_clause, values=values_as_json)


def format_correct_cluster_prompt(
        values: List[str],
        clusters: List[int],
        correct_answer: str,
        unit: str
) -> str:
    unit_suffix = f" {unit}" if unit and unit.strip() else ""
    clusters_string = format_clusters(values, clusters)
    return correct_cluster_prompt.format(
        clusters=clusters_string,
        correct_answer=correct_answer,
        unit=unit_suffix
    )


def format_clusters(values: List[str], cluster_ids: List[Optional[int]]) -> str:
    if len(values) != len(cluster_ids):
        min_length = min(len(values), len(cluster_ids))
        values = values[:min_length]
        cluster_ids = cluster_ids[:min_length]
    clusters = defaultdict(list)
    for value, cluster_id in zip(values, cluster_ids):
        if cluster_id is not None:
            clusters[cluster_id].append(value)
    parts = []
    for cluster_id in sorted(clusters):
        parts.append(f"Cluster with cluster_id {cluster_id}:")
        for value in clusters[cluster_id]:
            parts.append(f"* {value}")
        parts.append("")  # Blank line between clusters
    result = "\n".join(parts).strip()
    if len(result) == 0:
        return "No Clusters available, please directly return NONE!"
    return result


def _parse_cluster(cluster_str: str):
    return int(cluster_str.strip()) if cluster_str.strip().isdigit() else None


def _parse_simplified_and_clusters(text: str, schema: Dict):
    try:
        parsed_dict = json.loads(text)
        jsonschema.validate(instance=parsed_dict, schema=schema)  # TODO sanity check, may be removed if it works out
        simplified_values = [item["value"] for item in parsed_dict["simplified_values"]]
        clusters = [_parse_cluster(item["cluster_id"]) for item in parsed_dict["cluster_ids"]]
        return simplified_values, clusters
    except jsonschema.ValidationError as e:
        print("Invalid:", e.message)
        raise


def parse_correct_cluster(response: str) -> int:
    matches = re.findall(r"\d+", response)
    if matches:
        return int(matches[-1])

    if "none" in response.lower():
        return -1

    raise ValueError(f"Could not parse correct cluster ID from response: {response!r}")


def cluster_freq_list(cluster_ids: List[int]) -> List[float]:
    n = len(cluster_ids)
    counts = Counter(cluster_ids)
    return [
        0.0 if cid is None else counts[cid] / n
        for cid in cluster_ids
    ]


class SciBenchAnswerFrequency:
    dependencies = ["sampled_conclusion_texts", "correct_answer", "unit", "sampled_assistant_tokens_decoded", "sampled_reasoning_tokens_decoded"]
    stats = ["cluster_id", "frequency_of_answer", "correct_cluster_id", "conclusion_texts_array", "is_correct",
             "simplified_values", "answer_token_len", "reasoning_token_len"]
    spread = True

    def __init__(self, sampling_n: int):
        self.json_schema = get_json_schema(sampling_n)

    async def __call__(self, stats: Dict[str, np.ndarray], model: Model) -> Dict[str, np.ndarray]:
        sampled = stats["sampled_conclusion_texts"]
        correct = stats["correct_answer"]
        unit = stats["unit"]
        n = len(sampled[0])

        def _clean(v: str) -> str:
            v = v.replace("<|eot_id|>", "").replace("\n", " ").strip()
            return v if v else "<NONE>"

        # clean out artifact markers
        sampled = [[_clean(v) for v in row] for row in sampled]

        # 1. clustering queries
        cluster_msgs = [
            [{"role": "user", "content": format_clustering_prompt(s, u)}]
            for s, u in zip(sampled, unit)
        ]

        try:
            params1 = GenerationParameters(
                max_tokens=2048,
                logprobs=0,
                temperature=0,
                seed=random.getrandbits(32),
                response_format={"type": "json_schema", "json_schema": self.json_schema}
            )
            resp1 = await model.query(cluster_msgs, generation_params=params1)
            msgs1 = resp1.get_assistant_messages()

            results = [_parse_simplified_and_clusters(m, self.json_schema) for m in msgs1]
            simplified_values = [sv for sv, _ in results]
            clusters = [c for _, c in results]
            assert len(simplified_values) == len(clusters), "Simplified values and clusters do not match!"
            for i in range(len(simplified_values)):
                assert isinstance(simplified_values[i], list) and len(simplified_values[
                                                                          i]) != 0, f"Simplified values for generation {i} is empty: {simplified_values[i]}!"
                assert isinstance(clusters[i], list) and len(
                    clusters[i]) != 0, f"Clusters for generation {i} is empty: {clusters[i]}!"
        except Exception as e:
            log.error(
                f"Error during parsing: {e}")
            raise

        # 2. correct-cluster queries as new conversations
        correct_msgs = [
            [{"role": "user", "content": format_correct_cluster_prompt(
                values=v, clusters=c, correct_answer=ca, unit=u
            )}]
            for v, c, ca, u in zip(simplified_values, clusters, correct, unit)
        ]

        params2 = GenerationParameters(max_tokens=400, logprobs=0, temperature=0.0,
                                       response_format={"type": "regex", "regex": "^(?:\\d{1,2}|NONE)$"})
        resp2 = await model.query(correct_msgs, generation_params=params2)
        msgs2 = resp2.get_assistant_messages()
        correct_clusters = [parse_correct_cluster(m) for m in msgs2]

        is_correct = [
            [el == correct_cluster for el in cluster]
            for cluster, correct_cluster in zip(clusters, correct_clusters)
        ]

        return {
            "cluster_id": clusters,
            "frequency_of_answer": [cluster_freq_list(c) for c in clusters],
            "correct_cluster_id": np.array(correct_clusters),
            "conclusion_texts_array": sampled,
            "is_correct": is_correct,
            "answer_token_len": [[len(t) for t in sample] for sample in stats["sampled_assistant_tokens_decoded"]],
            "reasoning_token_len": [[len(t) for t in sample] for sample in stats["sampled_reasoning_tokens_decoded"]],
            "simplified_values": simplified_values
        }
