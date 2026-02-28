from datasets import load_dataset
from tqdm import tqdm

def load_codesearchnet_subset(
    split="train",
    max_samples=1000
):
    dataset = load_dataset(
        "Nan-Do/code-search-net-python",
        split=split
    )

    data = []

    for example in tqdm(dataset):
        docstring = example["docstring"]
        code = example["code"]

        if not docstring or not code:
            continue

        data.append((docstring, code))

        if len(data) >= max_samples:
            break

    return data
