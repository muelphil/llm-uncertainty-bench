from collections import defaultdict


class AnsweredCorrectlyMC:
    requires = ["selected_options", "correct_answer",
                "sampled_assistant_tokens_decoded", "sampled_reasoning_tokens_decoded", "sampled_finish_reasons",
                "sampled_yes_no_probabilities", "sampled_model_labeled_option_correct"]
    provides = ["is_correct", "yes_no_probabilities", "correct_answer_idx", "answer_token_len", "reasoning_token_len",
                "model_guessed_its_correct", "frequency_of_answer", "finish_reasons"]
    spread = True

    def __call__(self, deps):
        result = defaultdict(list)
        for correct_ans, evaluated_ans, tokens_samples, reasoning_tokens_samples, model_guessed_its_correct, yes_no_probabilities, finish_reasons in zip(
                deps["correct_answer"],
                deps["selected_options"],
                deps["sampled_assistant_tokens_decoded"],
                deps["sampled_reasoning_tokens_decoded"],
                deps["sampled_model_labeled_option_correct"],
                deps["sampled_yes_no_probabilities"],
                deps["sampled_finish_reasons"]
        ):
            is_correct = [(correct_ans == evaluated_ans) == guessed_correct if guessed_correct is not None else False
                          for guessed_correct in model_guessed_its_correct]

            frequency_of_answer = [
                (model_guessed_its_correct.count(guessed_correct) / len(
                    model_guessed_its_correct)) if guessed_correct is not None else 0.0 for
                guessed_correct in model_guessed_its_correct
            ]

            result["is_correct"].append(is_correct)
            result["model_labeled_option_correct"].append(model_guessed_its_correct)
            result["yes_no_probabilities"].append(yes_no_probabilities)
            result["correct_answer_idx"].append(correct_ans)
            result["frequency_of_answer"].append(frequency_of_answer)
            result["answer_token_len"].append([len(tokens) for tokens in tokens_samples])
            result["reasoning_token_len"].append([len(tokens) for tokens in reasoning_tokens_samples])
            result["finish_reasons"].append(finish_reasons)
        return result
