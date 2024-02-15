import copy
import heapq
import argparse


class State:

    def __init__(self):
        self.board = [[0] * 3 for _ in range(3)]
        # g: current cost to reach node
        self.g = 0
        # h: heuristic estimated cost to goal
        self.h = 0
        # f: total cost (f=g+h)
        self.f = 0
        self.parent = None

    # less than comparison based on total cost
    def __lt__(self, other):
        return self.f < other.f


# Goal State for the puzzle
goal = [[7, 8, 1],
        [6, 0, 2],
        [5, 4, 3]]


# function to display the board to standard output
def print_board(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                print("*", end=' ')
            else:
                print(state[i][j], end=' ')
        print()
    print()


# finds empty tile position
def empty_pos(state):
    for i in range(3):
        for j in range(3):
            if state.board[i][j] == 0:
                return i, j


# expands node and gets successors (builds the state-space tree)
def expand(state):

    # fringe: list of nodes that have been generated but not yet expanded/visited
    fringe = []
    i, j = empty_pos(state)

    if i > 0:
        child = copy.deepcopy(state)
        child.board[i][j], child.board[i - 1][j] = child.board[i - 1][j], child.board[i][j]
        child.g = state.g + 1
        child.parent = state
        fringe.append(child)

    if i < 2:
        child = copy.deepcopy(state)
        child.board[i][j], child.board[i + 1][j] = child.board[i + 1][j], child.board[i][j]
        child.g = state.g + 1
        child.parent = state
        fringe.append(child)

    if j > 0:
        child = copy.deepcopy(state)
        child.board[i][j], child.board[i][j - 1] = child.board[i][j - 1], child.board[i][j]
        child.g = state.g + 1
        child.parent = state
        fringe.append(child)

    if j < 2:
        child = copy.deepcopy(state)
        child.board[i][j], child.board[i][j + 1] = child.board[i][j + 1], child.board[i][j]
        child.g = state.g + 1
        child.parent = state
        fringe.append(child)

    return fringe


# Heuristic 1 - Manhattan Distance
def manhattan(state):
    dist = 0

    for i in range(3):
        for j in range(3):
            val = state.board[i][j]
            if val != 0:
                x, y = (val - 1) // 3, (val - 1) % 3
                dist += abs(x - i) + abs(y - j)

    return dist


# Heuristic 2 - Number of Misplaced Tiles
def incorrect_positions(state, goalState):
    incorrect = 0
    for i in range(3):
        for j in range(3):
            if state.board[i][j] != 0 and state.board[i][j] != goalState[i][j]:
                incorrect += 1
    return incorrect


# Iterative deepening search
def ids(start):
    depth = 0
    nodes_expanded = 0

    while True:
        result, visited = dfs(start, depth)
        nodes_expanded += visited

        if result is not None or depth >= 10:
            return result, nodes_expanded

        depth += 1


# Depth-first search
def dfs(start, depth_limit):
    stack = [start]
    visited = set()

    while stack:
        state = stack.pop()

        if state.g > depth_limit:
            continue

        if hash(str(state.board)) in visited:
            continue

        visited.add(hash(str(state.board)))

        if state.board == goal:
            return state, len(visited)

        # build the state-space tree by expanding nodes
        successors = expand(state)
        for succ in successors:
            if hash(str(succ.board)) not in visited:
                stack.append(succ)

    return None, len(visited)


# Initialize closed set for A* search algos
closed = set()


# A* (1) search algorithm
def astar1(start):
    # print("Searching using A*...")

    fringe = [start]

    # turns the regular list into a min heap
    heapq.heapify(fringe)

    while fringe:
        state = heapq.heappop(fringe)

        if hash(str(state.board)) in closed:
            continue

        closed.add(hash(str(state.board)))

        if state.board == goal:
            # print("Path found!")
            return state, len(closed)

        # evaluate state using heuristic 1
        state.h = manhattan(state)
        state.f = state.g + state.h

        # build the state-space tree by expanding nodes
        successors = expand(state)
        for succ in successors:
            heapq.heappush(fringe, succ)

    return None, len(closed)


# A* (2) search algorithm
def astar2(start):
    # print("Searching using A* with Heuristic #2")

    fringe = [start]

    # turns the regular list into a min heap
    heapq.heapify(fringe)

    while fringe:
        state = heapq.heappop(fringe)

        if hash(str(state.board)) in closed:
            continue

        closed.add(hash(str(state.board)))

        if state.board == goal:
            # print("Path found!")
            return state, len(closed)

        # evaluate the state using heuristic 2
        state.h = incorrect_positions(state, goal)
        state.f = state.g + state.h

        # build the state-space tree by expanding nodes
        successors = expand(state)
        for succ in successors:
            if hash(str(succ.board)) not in closed:
                heapq.heappush(fringe, succ)

    return None, len(closed)


def process_file_return_matrix(file_path):
    try:
        with open(file_path, 'r') as file:

            matrix = []

            # Read single line
            line = file.readline().replace(" ", "")

            for i in range(3):
                # Extract 3 characters at a time (one row of the 3x3 matrix)
                row = line[3 * i:3 * i + 3]

                # Convert to ints, and convert '*' to 0
                row = [int(x) if x != '*' else 0 for x in row]

                matrix.append(row)

            # print(matrix)

            return matrix

    # handle file related errors
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # get arguments from terminal input
    parser = argparse.ArgumentParser()
    parser.add_argument("str", type=str, help="Input for algo selection")
    parser.add_argument("file_path", help="Path to input file")

    args = parser.parse_args()

    # process file input and algorithm selection
    start = State()
    start.board = process_file_return_matrix(args.file_path)
    algo_selection = args.str

    print("\n\nInitial Puzzle State:")
    print_board(start.board)
    print("\n")

    # branching logic to run the correct search algorithm based on user input from terminal
    goal_state = False
    nodes_expanded = 0

    if algo_selection == "dfs":
        # print("dfs was selected")
        goal_state, nodes_expanded = dfs(start, 10)

    elif algo_selection == "ids":
        # print("ids was selected")
        goal_state, nodes_expanded = ids(start)

    elif algo_selection == "astar1":
        # print("astar1 was selected")
        goal_state, nodes_expanded = astar1(start)

    elif algo_selection == "astar2":
        # print("astar2 was selected")
        goal_state, nodes_expanded = astar2(start)

    else:
        print("INPUT_ERROR: Please selected an algorithm from the following list...\n")
        print("\tdfs")
        print("\tids")
        print("\tastar1")
        print("\tastar2\n\n")

    if goal_state:
        print("Path:")
        path = []
        while goal_state:
            path.append(goal_state)
            goal_state = goal_state.parent

        for state in reversed(path):
            print_board(state.board)

        print("Number of moves:", len(path) - 1)
    else:
        print("No solution found.")

    print("Number of states enqueued:", nodes_expanded)


if __name__ == "__main__":
    main()
