import re

WHITESPACE_TOKENS_REGEX = re.compile(r"[Ġ▁]")


def decode_if_byte(text: str):
    return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text


def sanitize_token(text: str) -> str:
    return (text.replace("Ċ", "\n")
            .replace("▁", " ")
            .replace("Ġ", " ")
            .replace("<｜end of sentence｜>", "")
            .replace("<|im_end|>","")
            .replace('<|eot_id|>', ''))


class ResponseCompressorSerializer:  # TODO move to async-graph-bench

    def __init__(self, prefix="greedy"):
        self.prefix = prefix
        self.computable_keys = {
            "texts": prefix + "_texts",
            "log_probs": prefix + "_log_probs",
            "log_likelihoods": prefix + "_log_likelihoods",
            "tokens_decoded": prefix + "_tokens_decoded",
        }

    def serialize(self, item: dict) -> dict:
        if self.prefix + '_tokens_decoded_alternatives' not in item:
            return item
        removed_keys = [key for key in self.computable_keys.values() if key in item]

        serialized = {key: value for key, value in item.items() if key not in removed_keys}
        if not "_removed_keys" in serialized:
            serialized["_removed_keys"] = removed_keys
        else:
            serialized["_removed_keys"] += removed_keys

        return serialized

    def deserialize(self, data: dict) -> dict:

        if '_removed_keys' not in data:
            return data

        alternatives_per_token = data[f"{self.prefix}_tokens_decoded_alternatives"]
        alternatives_per_token = [[[sanitize_token(decode_if_byte(token)), score] for token, score in sublist] for sublist in
                                  alternatives_per_token]  # this is necessary because sentencepiece in its more
        data[f"{self.prefix}_tokens_decoded_alternatives"] = alternatives_per_token
        removed_keys = data['_removed_keys']

        if self.computable_keys["texts"] in removed_keys:
            data[self.computable_keys["texts"]] = "".join(
                [alternatives[0][0] for alternatives in alternatives_per_token])
            if data[self.computable_keys["texts"]].endswith("<|eot_id|>"):
                data[self.computable_keys["texts"]] = data[self.computable_keys["texts"]][0: -len("<|eot_id|>")]

        if self.computable_keys["log_probs"] in removed_keys:
            data[self.computable_keys["log_probs"]] = [[alt[1] for alt in alternatives] for alternatives in
                                                       alternatives_per_token]

        if self.computable_keys["log_likelihoods"] in removed_keys:
            data[self.computable_keys["log_likelihoods"]] = [alternatives[0][1] for alternatives in
                                                             alternatives_per_token]

        if self.computable_keys["tokens_decoded"] in removed_keys:
            data[self.computable_keys["tokens_decoded"]] = [alternatives[0][0] for alternatives in
                                                            alternatives_per_token]

        for item in self.computable_keys.values():
            if item in removed_keys:
                removed_keys.remove(item)

        if not removed_keys:
            del data['_removed_keys']

        return data
