class PartialResponseDeserializer:  # TODO move to async-graph-bench
    def serialize(self, item: dict) -> dict:
        raise NotImplementedError()

    def deserialize(self, data: dict) -> dict:
        alternatives_per_token = data["greedy_tokens_decoded_alternatives"]
        data['greedy_texts'] = "".join([alternatives[0][0] for alternatives in alternatives_per_token])
        for key in ['_removed_keys', 'greedy_tokens_decoded_alternatives', 'conclusion_decoded_alternatives',
                    'conclusion_log_probs']:
            if key in data:
                del data[key]

        return data
