import ast
import json
import re
import sys
from pprint import pprint


def load_data(dataset: str, which: str) -> dict:
    with open(f'{dataset}-{which}-analysis.txt') as f:
        data = f.read()

    data = re.sub(r"(\w)'(\w)", r"\1\\'\2", data)
    return ast.literal_eval(data)


def get_difference_between_with_without_mr(dataset: str):
    with_mr = load_data(dataset, "mr")
    without_mr = load_data(dataset, "basic")

    keys = with_mr.keys()
    diffs = []
    for key in keys:
        diffs.append(abs(without_mr[key] - with_mr[key]))

    return diffs

if __name__ == '__main__':
    dataset = "dolphins"
    diffs = get_difference_between_with_without_mr(dataset)
    average = sum(diffs) / len(diffs)

    with open(f"{dataset}-diffs.txt", 'w') as sys.stdout:
        print(f"{average=}")
        pprint(f"{diffs=}")


