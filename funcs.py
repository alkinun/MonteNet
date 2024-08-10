
# This function flips all colors in the grid (changes 0 to 1 and vice versa)
def flip_colors(grid):
    return [[1 if cell == 0 else 0 for cell in row] for row in grid]

# This function counts the number of 1s around each cell in the grid
def count_ones(grid, i, j):
    total = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            if (x != 0 or y != 0) and 0 <= i+x < len(grid) and 0 <= j+y < len(grid[0]):
                total += grid[i+x][j+y]
    return total

# This function generates a new grid where each cell is the count of ones around it in the original grid
def create_count_grid(grid):
    return [[count_ones(grid, i, j) for j in range(len(grid[0]))] for i in range(len(grid))]

# This function changes all cells with a certain value to 1 and the rest to 0
def threshold(grid, val):
    return [[1 if cell >= val else 0 for cell in row] for row in grid]

# This function shifts the grid one step to the right
def shift_right(grid):
    return [row[-1:]+row[:-1] for row in grid]

# This function shifts the grid one step down
def shift_down(grid):
    return grid[1:] + grid[:1]

# The rest of the functions are similar but perform different transformations on the grid.
# For example, you could have a function that shifts the grid left or up, a function that flips the grid horizontally or vertically, etc.
