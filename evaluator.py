import os
import evaluate
from typing import List

os.environ["HUGGINGFACE_HUB_CACHE"] = (
    "/scratch/" + os.environ["USER"] + "/huggingface_cache"
)

os.environ["TRANSFORMERS_CACHE"] = (
    "/scratch/" + os.environ["USER"] + "/huggingface_cache"
)

CACHE_DIR = "/scratch/" + os.environ["USER"] + "/huggingface_cache"


def perplexity_evaluator(preds: List[str]):
    """
    Given a model and an input text sequence, perplexity measures how likely the model is to generate the input text sequence.

    Lower is better.
    """
    perplexity = evaluate.load("perplexity", module_type="metric", cache_dir=CACHE_DIR)

    results = perplexity.compute(predictions=preds, model_id="microsoft/phi-2")
    return results["perplexities"]


def charcut_evaluator(ref: str, preds: List[str]):
    """
    The matching algorithm is based on an iterative search for longest common substrings, combined with a length-based threshold that limits short and noisy character matches.

    Lower is better.
    """
    print("Evaluating charcut... Higher is better.")
    refs = [ref] * len(preds)
    charcut = evaluate.load("charcut_mt", module_type="metric", cache_dir=CACHE_DIR)

    results = charcut.compute(references=refs, predictions=preds)
    print(results)
    return None


if __name__ == "__main__":
    preds = ["This is a test", "This is another test"]
    ref = "This is a test"
    # results = perplexity_evaluator(preds)
    # print(results)
    results = charcut_evaluator(ref, preds)
    print(results)
