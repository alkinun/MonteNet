from llm import main as llm
from dps import main as dps
from data import examples

def main(examples):
    for i in range(1):
        intuition, code = llm(examples)
        with open(f"funcs.py", "w") as f:
            f.write(code)
        algo = dps(examples)
        if algo:
            return algo
        else:
            print("No algorithm found")

print(main(examples))
