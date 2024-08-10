import ollama
from data import examples
import json


def llm(messages, model):
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]

def encode_grid(grid):
    return "```json\n" + json.dumps(grid).replace("],", '],\n') + "\n```"

def create_prompt_intution(examples):
    intuition_prompt = """You are a helpful AI assistant. Your job is to provide intuitions about tasks from the Abstraction and Reasoning Challenge (ARC).
The user will present you with sample input and output grids for each task.
Your job will be to provide intuitions about the transformations between input and output, NO EXPLANATION OR RULE.
The puzzle-like inputs and outputs present a grid where each square can be one of ten colors. A grid can be any height or width between 1x1 and 30x30.
The background of the grid is typically colored with 0.
The transformations between input and output should be based on an underlying rule.
Remember that all the examples follow the same rule, your job is to provide intuitions about the transformations.
Reason about the transformations between input and output by looking at all the examples.
You always provide correct and detailed examinations and rule.
Here are some examples of transformations, they have the same rule:

"""
    for input_grid, output_grid in examples:
        intuition_prompt += f"#Example {examples.index((input_grid, output_grid))+1}\n##Input:\n{encode_grid(input_grid)}\n\n##Output:\n{encode_grid(output_grid)}\n\n---\n\n"
    intuition_prompt += "Now, provide intuitions about the transformations between input and output, DO NOT PROVIDE ANY EXPLANATION, RULE OR COMPLETE ANSWER, just intuitions and toughts.\nRemember that all the examples respect the exact same rule.\nThink step by step, and reason in a really detailed manner."
    return intuition_prompt

def create_prompt_code():
    code_prompt = f"""Now, generate 20 small Python functions that can be useful when solving the task.
Do not provide a complete function that can solve the task at once, but rather provide multiple, very simple transformation functions that can be used when solving the task.
Here is an exact template function that you will build on top of:
```python
# Describe the function here
def function_name(grid):
    # Your code here
    return output
```

**Important:**
- All functions should only have one argument: `grid` and no other arguments.
- Each function should only return a modified grid.
- Do NOT include any additional arguments or parameters besides `grid` in all functions.
- Each function should be a small, single-purpose transformation related to the task, not a complete solution.
- Ensure the functions are independent and can be combined together to solve parts of the task.
- Write all the one arg functions in a single ```python ``` block.

---

Now, write 20 new tiny functions related to the task. Remember, all functions must take only one argument: `grid`, and output a grid. Do NOT include functions with more than one argument."""
    return code_prompt

def main(examples):
    prompt = create_prompt_intution(examples)
    messages = [{"role": "user", "content": prompt}]
    intuition = llm(messages, "codestral")
    messages.append({"role": "assistant", "content": intuition})

    prompt = create_prompt_code()
    messages.append({"role": "user", "content": prompt})
    code = llm(messages, "codestral")
    code = code.split("```python")[1].split("```")[0]

    print(intuition)
    print("--------------------------------")
    print(code)

    return intuition, code
