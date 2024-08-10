import itertools
import funcs
from copy import deepcopy

def get_all_functions():
    functions = [getattr(funcs, name) for name in dir(funcs) if callable(getattr(funcs, name))]
    return functions

def generate_algorithms(functions, max_length=1):
    algorithms = []
    for length in range(1, max_length + 1):
        algorithms.extend(itertools.product(functions, repeat=length))
    return algorithms

def test_algorithm(algorithm, examples):
    for input_grid, expected_output in examples:
        modified_grid = deepcopy(input_grid)  # Create a deep copy to avoid in-place modifications
        for func in algorithm:
            try:
                modified_grid = func(modified_grid)
            except:
                return False
        if modified_grid != expected_output:
            return False
    return True

def main(examples):
    functions = get_all_functions()
    algorithms = generate_algorithms(functions)

    for i, algorithm in enumerate(algorithms):
        if test_algorithm(algorithm, examples):
            return [func.__name__ for func in algorithm]
    else:
        return None
