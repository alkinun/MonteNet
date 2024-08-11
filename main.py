from llm import main as llm
from dps import main as dps
from data import examples
from data import input_grid
import re

def main(examples):
    for i in range(3):
        intuition, code = llm(examples)

        with open(f"funcs.py", "w") as f:
            f.write(code)

        algo = dps(examples)

        with open(f"funcs.py", "w") as f:
            f.write("")

        if algo:
            match = re.search(r'def (\w+)\(', algo[0])
            if match:
                func_name = match.group(1)
                exec(algo[0], globals())
                output = globals()[func_name](input_grid)
                return output
        else:
            print("Nope")

print(main(examples))
