#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <chrono>
#include <tuple> 


// Sequential single maze generation took 199877 ms
// 101 * 101 * 180 * 180 = equivalent to 18180x18180 maze (18181 since it needs to be odd)
// 101 * 101 * 180 * 180 * 4 Bytes = 1,322,049,600 Bytes = 1.23 GB

// 21 * 21 * 3 * 3 = equivalent to 63x63 maze

#define N 101  // Size of individual mazes (N x N)
#define P 180 // Number of mazes in one row/column of the large maze
#define MAX_SIZE (N * N)
#define LARGE_MAZE_SIZE (N * P)  // Size of the large maze (N*P x N*P)

#define cudaCheckError() {                               \
    cudaError_t e = cudaGetLastError();                    \
    if (e != cudaSuccess) {                                \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                \
    }                                                     \
}

enum MAZE_PATH {
    EMPTY = 0x0,
    WALL = 0x1,
    EXIT = 0x2,
    SOLUTION = 0x3,
    START = 0x4,
    PARTICLE = 0x5,
};

#include <vector>

// Union-Find data structure to manage connected components
struct UnionFind {
    std::vector<int> parent;
    std::vector<int> rank;

    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    bool union_sets(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            }
            else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            }
            else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            return true;
        }
        return false;
    }
};


// Union-Find structure for GPU
struct UnionFindGPU {
    int* parent;
    int* rank;

    __device__ UnionFindGPU(int n, int* parentArray, int* rankArray) {
        parent = parentArray;
        rank = rankArray;
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    __device__ int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    __device__ bool union_sets(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            }
            else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            }
            else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            return true;
        }
        return false;
    }
};

__device__ void initialize_maze_cuda(MAZE_PATH* maze, int size, int* exit_row, int* exit_col, curandState* localState);
__global__ void init_rng(curandState* state, unsigned long seed);
__global__ void generate_mazes(curandState* globalState, MAZE_PATH* mazes);
__global__ void combine_mazes(MAZE_PATH* small_mazes, MAZE_PATH* large_maze, int small_size, int large_size, int num_mazes_per_side);

__device__ void dfs_maze_generation(MAZE_PATH* maze, int size, int start_row, int start_col, curandState* localState);
__device__ void print_maze_thread(MAZE_PATH* maze, int size);

void connect_mazes(MAZE_PATH* large_maze, int num_mazes, int small_size, int large_size, UnionFind& uf);
void print_combined_maze(MAZE_PATH* large_maze, int large_size);
void combine_mazes_cpu(MAZE_PATH* small_mazes, MAZE_PATH* large_maze);

__device__ void print_maze_thread(MAZE_PATH* maze, int size) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            switch (maze[row * size + col]) {
            case MAZE_PATH::EMPTY:
                printf(" ");
                break;
            case MAZE_PATH::WALL:
                printf("#");
                break;
            case MAZE_PATH::EXIT:
                printf("E");
                break;
            case MAZE_PATH::SOLUTION:
                printf(".");
                break;
            case MAZE_PATH::START:
                printf("S");
                break;
            case MAZE_PATH::PARTICLE:
                printf("P");
                break;
            default:
                printf("?");
                break;
            }
            printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

// Function to print the combined maze
void print_combined_maze(MAZE_PATH* large_maze, int large_size) {
    for (int row = 0; row < large_size; ++row) {
        for (int col = 0; col < large_size; ++col) {
            switch (large_maze[row * large_size + col]) {
            case MAZE_PATH::EMPTY:
                std::cout << " ";
                break;
            case MAZE_PATH::WALL:
                std::cout << "#";
                break;
            case MAZE_PATH::EXIT:
                std::cout << "E";
                break;
            case MAZE_PATH::SOLUTION:
                std::cout << ".";
                break;
            case MAZE_PATH::START:
                std::cout << "S";
                break;
            case MAZE_PATH::PARTICLE:
                std::cout << "P";
                break;
            default:
                std::cout << "?";
                break;
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }
}

// Kernel function to initialize random states
__global__ void init_rng(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed + idx, idx, 0, &state[idx]);

    // Debug print to ensure that initialization happened
    if (idx >= 0) {  // Only print for the first thread to reduce clutter
        //printf("curand state initialized for idx = %d\n", idx);
        // Test random number generation
        float random_number = curand_uniform(&state[idx]);
        //printf("Random number generated by thread %d: %f\n", idx, random_number);
    }
}

// GPU function to initialize the maze
__device__ void initialize_maze_cuda(MAZE_PATH* maze, int size, int* exit_row, int* exit_col, curandState* localState) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            if (row % 2 == 0 || col % 2 == 0) {
                maze[row * size + col] = MAZE_PATH::WALL;
            }
            else {
                maze[row * size + col] = MAZE_PATH::EMPTY;
            }
        }
    }

    // Randomly choose a border for the exit (0: top/bottom, 1: left/right)
    int border_choice = curand(localState) % 2;

    if (border_choice == 0) {
        *exit_row = (curand(localState) % 2 == 0) ? 0 : size - 1;
        *exit_col = (curand(localState) % (size / 2)) * 2 + 1;
    }
    else {
        *exit_row = (curand(localState) % (size / 2)) * 2 + 1;
        *exit_col = (curand(localState) % 2 == 0) ? 0 : size - 1;
    }

    maze[*exit_row * size + *exit_col] = MAZE_PATH::EXIT;
}

// DFS Maze Generation with Backtracking
__device__ void dfs_maze_generation(MAZE_PATH* maze, int size, int start_row, int start_col, curandState* localState) {
    // Stack-based DFS using explicit stack
    int stack[MAX_SIZE][2]; // Each entry holds (row, col)
    int stack_size = 0;

    bool visited[MAX_SIZE] = { false };  // Track visited cells
    int visited_count = 1;  // Start with the initial cell

    stack[stack_size][0] = start_row;
    stack[stack_size][1] = start_col;
    visited[start_row * size + start_col] = true;
    stack_size++;

    // Direction vectors for Up, Down, Left, Right
    int direction[4][2] = { {-2, 0}, {2, 0}, {0, -2}, {0, 2} };
    int exit_direction[4][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };  // Direction to exit from the start

    int total_cells = ((size - 1) / 2) * ((size - 1) / 2) + 1; // Calculate total cells in the maze that can be visited (paths, not walls)
    int iteration_count = 0;  // Counter to prevent infinite loops
    bool is_first_move = true;

    while (stack_size > 0) {
        int curr_row = stack[stack_size - 1][0];
        int curr_col = stack[stack_size - 1][1];

        int directions_to_try[4] = { 0, 1, 2, 3 };

        // Shuffle directions to introduce randomness
        for (int i = 3; i > 0; --i) {
            int j = curand(localState) % (i + 1);
            int temp = directions_to_try[i];
            directions_to_try[i] = directions_to_try[j];
            directions_to_try[j] = temp;
        }

        bool path_found = false;

        // Explore all possible neighbors in random order
        for (int i = 0; i < 4; ++i) {
            int new_row, new_col;

            if (is_first_move) {
                // For the first move from the exit, use exit_direction to move away from the boundary
                new_row = curr_row + exit_direction[directions_to_try[i]][0];
                new_col = curr_col + exit_direction[directions_to_try[i]][1];
            }
            else {
                // After the first move or if not starting from the exit, use normal direction vectors
                new_row = curr_row + direction[directions_to_try[i]][0];
                new_col = curr_col + direction[directions_to_try[i]][1];
            }

            // Ensure we're not moving out of bounds
            if (new_row >= 1 && new_row < size - 1 && new_col >= 1 && new_col < size - 1) {
                // Ensure the cell hasn't been visited before
                if (!visited[new_row * size + new_col]) {
                    visited[new_row * size + new_col] = true;
                    visited_count++;

                    // Remove the wall between the current and new cell
                    int wall_row = (curr_row + new_row) / 2;
                    int wall_col = (curr_col + new_col) / 2;
                    maze[wall_row * size + wall_col] = MAZE_PATH::EMPTY;

                    // Push the new cell onto the stack
                    stack[stack_size][0] = new_row;
                    stack[stack_size][1] = new_col;
                    stack_size++;

                    path_found = true;
                    is_first_move = false;  // Reset after the first move
                    break;
                }
            }
        }

        // If no path was found, backtrack
        if (!path_found) {
            stack_size--;  // Pop from stack
        }

        // If all cells are visited, stop the DFS
        if (visited_count >= total_cells) {
            break;
        }

        iteration_count++;
        if (iteration_count > MAX_SIZE * 10) {
            // Exit if too many iterations to prevent infinite loop
            printf("Thread %d: Exiting after too many iterations\n", blockIdx.x * blockDim.x + threadIdx.x);
            break;
        }

        //maze[start_row * size + start_col] = MAZE_PATH::EMPTY; // mark the exit as 0
        maze[start_row * size + start_col] = MAZE_PATH::WALL; // close the exit
    }
}

// Kernel function to generate individual mazes
__global__ void generate_mazes(curandState* globalState, MAZE_PATH* mazes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get the random state for this thread
    curandState localState = globalState[idx];

    // Pointer to this thread's maze in the global memory
    MAZE_PATH* maze = &mazes[idx * N * N];

    int exit_row, exit_col;

    // Initialize the maze
    initialize_maze_cuda(maze, N, &exit_row, &exit_col, &localState);

    // Generate the maze paths using DFS with backtracking
    dfs_maze_generation(maze, N, exit_row, exit_col, &localState);

    // Store the updated state back to global memory
    globalState[idx] = localState;
}

// Sequential CPU function to combine individual mazes into a large maze
void combine_mazes_cpu(MAZE_PATH* small_mazes, MAZE_PATH* large_maze) {
    // Iterate over each maze block
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            // Copy the individual maze into the correct position in the large maze
            for (int row = 0; row < N; ++row) {
                for (int col = 0; col < N; ++col)
                    large_maze[(i * N + row) * LARGE_MAZE_SIZE + (j * N + col)] =
                    small_mazes[(i * P + j) * MAX_SIZE + row * N + col];
            }
        }
    }
}




// Kernel function to combine individual mazes into a large maze
__global__ void combine_mazes(MAZE_PATH* small_mazes, MAZE_PATH* large_maze, int small_size, int large_size, int num_mazes_per_side) {
    int small_row = blockIdx.y * blockDim.y + threadIdx.y;
    int small_col = blockIdx.x * blockDim.x + threadIdx.x;
    int maze_index = blockIdx.z;

    if (small_row < small_size && small_col < small_size) {
        int large_row = (maze_index / num_mazes_per_side) * small_size + small_row;
        int large_col = (maze_index % num_mazes_per_side) * small_size + small_col;

        large_maze[large_row * large_size + large_col] = small_mazes[maze_index * small_size * small_size + small_row * small_size + small_col];
    }
}


// Function to connect neighboring mazes
void connect_mazes(MAZE_PATH* large_maze, UnionFind& uf) {
    int total_mazes = P * P;

    std::vector<std::pair<int, int>> directions = {
        {0, -1}, {0, 1}, {-1, 0}, {1, 0}
    };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> maze_dist(0, total_mazes - 1);
    std::uniform_int_distribution<> dir_dist(0, static_cast<int>(directions.size()) - 1);

    // Randomly connect neighboring mazes
    while (uf.find(0) != uf.find(total_mazes - 1)) {
        int maze_a = maze_dist(gen);
        int dir_idx = dir_dist(gen);

        int row_a = maze_a / P;
        int col_a = maze_a % P;

        int row_b = row_a + directions[dir_idx].first;
        int col_b = col_a + directions[dir_idx].second;

        // Ensure the neighboring maze is within bounds
        if (row_b >= 0 && row_b < P && col_b >= 0 && col_b < P) {
            int maze_b = row_b * P + col_b;

            // If these mazes are not already connected, connect them
            if (uf.union_sets(maze_a, maze_b)) {
                // Calculate the wall positions to remove
                int wall_row_a = row_a * N + N / 2 + directions[dir_idx].first * (N / 2);
                int wall_col_a = col_a * N + N / 2 + directions[dir_idx].second * (N / 2);

                int wall_row_b = wall_row_a + directions[dir_idx].first;
                int wall_col_b = wall_col_a + directions[dir_idx].second;

                // Remove the wall between the two mazes
                large_maze[wall_row_a * LARGE_MAZE_SIZE + wall_col_a] = MAZE_PATH::EMPTY;
                large_maze[wall_row_b * LARGE_MAZE_SIZE + wall_col_b] = MAZE_PATH::EMPTY;

                // Clear an additional cell adjacent to the connection point
                if (directions[dir_idx].first != 0) {  // Vertical connection
                    if (directions[dir_idx].first == -1) {  // maze_b is above maze_a
                        int clear_row_a = wall_row_a + 1;
                        int clear_row_b = wall_row_b - 1;
                        large_maze[clear_row_a * LARGE_MAZE_SIZE + wall_col_a] = MAZE_PATH::EMPTY;
                        large_maze[clear_row_b * LARGE_MAZE_SIZE + wall_col_b] = MAZE_PATH::EMPTY;
                    }
                    else {  // maze_b is below maze_a
                        int clear_row_a = wall_row_a - 1;
                        int clear_row_b = wall_row_b + 1;
                        large_maze[clear_row_a * LARGE_MAZE_SIZE + wall_col_a] = MAZE_PATH::EMPTY;
                        large_maze[clear_row_b * LARGE_MAZE_SIZE + wall_col_b] = MAZE_PATH::EMPTY;
                    }
                }
                else {  // Horizontal connection
                    if (directions[dir_idx].second == -1) {  // maze_b is to the left of maze_a
                        int clear_col_a = wall_col_a + 1;
                        int clear_col_b = wall_col_b - 1;
                        large_maze[wall_row_a * LARGE_MAZE_SIZE + clear_col_a] = MAZE_PATH::EMPTY;
                        large_maze[wall_row_b * LARGE_MAZE_SIZE + clear_col_b] = MAZE_PATH::EMPTY;
                    }
                    else {  // maze_b is to the right of maze_a
                        int clear_col_a = wall_col_a - 1;
                        int clear_col_b = wall_col_b + 1;
                        large_maze[wall_row_a * LARGE_MAZE_SIZE + clear_col_a] = MAZE_PATH::EMPTY;
                        large_maze[wall_row_b * LARGE_MAZE_SIZE + clear_col_b] = MAZE_PATH::EMPTY;
                    }
                }

                //printf("Connected maze %d and %d at grid (%d, %d) and (%d, %d) with positions (%d, %d) and (%d, %d) in the large maze.\n",
                //    maze_a, maze_b, row_a, col_a, row_b, col_b, wall_row_a, wall_col_a, wall_row_b, wall_col_b);
            }
        }
        else {
            //printf("Skipping invalid boundary connection between maze %d and outside boundaries.\n", maze_a);
        }
    }

    // Final pass to ensure all mazes are connected
    for (int maze = 0; maze < total_mazes; ++maze) {
        if (uf.find(maze) != uf.find(0)) {
            // Find a neighboring maze to connect
            for (const auto& dir : directions) {
                int row_b = (maze / P) + dir.first;
                int col_b = (maze % P) + dir.second;

                if (row_b >= 0 && row_b < P && col_b >= 0 && col_b < P) {
                    int maze_b = row_b * P + col_b;
                    if (uf.union_sets(maze, maze_b)) {
                        // Calculate the wall positions to remove
                        int wall_row_a = (maze / P) * N + N / 2 + dir.first * (N / 2);
                        int wall_col_a = (maze % P) * N + N / 2 + dir.second * (N / 2);

                        int wall_row_b = wall_row_a + dir.first;
                        int wall_col_b = wall_col_a + dir.second;

                        // Remove the wall between the two mazes
                        large_maze[wall_row_a * LARGE_MAZE_SIZE + wall_col_a] = MAZE_PATH::EMPTY;
                        large_maze[wall_row_b * LARGE_MAZE_SIZE + wall_col_b] = MAZE_PATH::EMPTY;

                        // Clear an additional cell adjacent to the connection point
                        if (dir.first != 0) {  // Vertical connection
                            if (dir.first == -1) {  // maze_b is above maze_a
                                int clear_row_a = wall_row_a + 1;
                                int clear_row_b = wall_row_b - 1;
                                large_maze[clear_row_a * LARGE_MAZE_SIZE + wall_col_a] = MAZE_PATH::EMPTY;
                                large_maze[clear_row_b * LARGE_MAZE_SIZE + wall_col_b] = MAZE_PATH::EMPTY;
                            }
                            else {  // maze_b is below maze_a
                                int clear_row_a = wall_row_a - 1;
                                int clear_row_b = wall_row_b + 1;
                                large_maze[clear_row_a * LARGE_MAZE_SIZE + wall_col_a] = MAZE_PATH::EMPTY;
                                large_maze[clear_row_b * LARGE_MAZE_SIZE + wall_col_b] = MAZE_PATH::EMPTY;
                            }
                        }
                        else {  // Horizontal connection
                            if (dir.second == -1) {  // maze_b is to the left of maze_a
                                int clear_col_a = wall_col_a + 1;
                                int clear_col_b = wall_col_b - 1;
                                large_maze[wall_row_a * LARGE_MAZE_SIZE + clear_col_a] = MAZE_PATH::EMPTY;
                                large_maze[wall_row_b * LARGE_MAZE_SIZE + clear_col_b] = MAZE_PATH::EMPTY;
                            }
                            else {  // maze_b is to the right of maze_a
                                int clear_col_a = wall_col_a - 1;
                                int clear_col_b = wall_col_b + 1;
                                large_maze[wall_row_a * LARGE_MAZE_SIZE + clear_col_a] = MAZE_PATH::EMPTY;
                                large_maze[wall_row_b * LARGE_MAZE_SIZE + clear_col_b] = MAZE_PATH::EMPTY;
                            }
                        }

                        //printf("Forced connection between maze %d and %d at grid (%d, %d) and (%d, %d) in final pass.\n",
                        //    maze, maze_b, maze / P, maze % P, row_b, col_b);
                        break;
                    }
                }
            }
        }
    }
}

// Sequential function to initialize the maze on the CPU
void initialize_maze_cpu(std::vector<MAZE_PATH>& maze, int size, int& exit_row, int& exit_col, std::mt19937& rng) {
    maze.resize(size * size, MAZE_PATH::WALL);

    for (int row = 1; row < size; row += 2) {
        for (int col = 1; col < size; col += 2) {
            maze[row * size + col] = MAZE_PATH::EMPTY;
        }
    }

    // Randomly choose a border for the exit (0: top/bottom, 1: left/right)
    std::uniform_int_distribution<int> border_choice_dist(0, 1);
    std::uniform_int_distribution<int> coord_dist(0, (size / 2) - 1);

    int border_choice = border_choice_dist(rng);

    if (border_choice == 0) {
        exit_row = (coord_dist(rng) * 2 + 1);
        exit_col = (rng() % 2 == 0) ? 0 : size - 1;
    }
    else {
        exit_row = (rng() % 2 == 0) ? 0 : size - 1;
        exit_col = (coord_dist(rng) * 2 + 1);
    }

    maze[exit_row * size + exit_col] = MAZE_PATH::EXIT;
}

// Sequential DFS Maze Generation with Backtracking on the CPU
void dfs_maze_generation_cpu(std::vector<MAZE_PATH>& maze, int size, int start_row, int start_col, std::mt19937& rng) {
    std::vector<std::pair<int, int>> stack;
    stack.emplace_back(start_row, start_col);

    std::vector<bool> visited(size * size, false);
    visited[start_row * size + start_col] = true;

    int direction[4][2] = { {-2, 0}, {2, 0}, {0, -2}, {0, 2} };
    int exit_direction[4][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };

    bool is_first_move = true;

    while (!stack.empty()) {
        // Unpack the current row and column from the stack
        int curr_row = stack.back().first;
        int curr_col = stack.back().second;
        stack.pop_back();

        std::vector<int> directions_to_try = { 0, 1, 2, 3 };
        std::shuffle(directions_to_try.begin(), directions_to_try.end(), rng);

        bool path_found = false;

        for (int i : directions_to_try) {
            int new_row, new_col;

            if (is_first_move) {
                new_row = curr_row + exit_direction[i][0];
                new_col = curr_col + exit_direction[i][1];
            }
            else {
                new_row = curr_row + direction[i][0];
                new_col = curr_col + direction[i][1];
            }

            if (new_row > 0 && new_row < size - 1 && new_col > 0 && new_col < size - 1) {
                if (!visited[new_row * size + new_col]) {
                    visited[new_row * size + new_col] = true;

                    int wall_row = (curr_row + new_row) / 2;
                    int wall_col = (curr_col + new_col) / 2;
                    maze[wall_row * size + wall_col] = MAZE_PATH::EMPTY;

                    stack.emplace_back(new_row, new_col);
                    path_found = true;
                    is_first_move = false;
                    break;
                }
            }
        }

        if (!path_found) {
            is_first_move = true;  // Allow backtracking to reconsider directions
        }
    }

    // Additional pass to ensure all cells are visited
    for (int i = 1; i < size; i += 2) {
        for (int j = 1; j < size; j += 2) {
            if (!visited[i * size + j]) {
                stack.emplace_back(i, j);
                visited[i * size + j] = true;
                is_first_move = true;
                while (!stack.empty()) {
                    int curr_row = stack.back().first;
                    int curr_col = stack.back().second;
                    stack.pop_back();

                    std::vector<int> directions_to_try = { 0, 1, 2, 3 };
                    std::shuffle(directions_to_try.begin(), directions_to_try.end(), rng);

                    for (int dir : directions_to_try) {
                        int new_row = curr_row + direction[dir][0];
                        int new_col = curr_col + direction[dir][1];

                        if (new_row > 0 && new_row < size - 1 && new_col > 0 && new_col < size - 1) {
                            if (!visited[new_row * size + new_col]) {
                                visited[new_row * size + new_col] = true;

                                int wall_row = (curr_row + new_row) / 2;
                                int wall_col = (curr_col + new_col) / 2;
                                maze[wall_row * size + wall_col] = MAZE_PATH::EMPTY;

                                stack.emplace_back(new_row, new_col);
                                is_first_move = false;
                                break;
                            }
                        }
                    }

                    if (!is_first_move) {
                        is_first_move = true;
                    }
                }
            }
        }
    }
}

// Kernel to generate mazes using Kruskal's algorithm
__global__ void generate_mazes_kruskal(curandState* globalState, MAZE_PATH* mazes, int* parentArray, int* rankArray, int* edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = globalState[idx];

    MAZE_PATH* maze = &mazes[idx * N * N];
    int* parent = &parentArray[idx * (N / 2) * (N / 2)];
    int* rank = &rankArray[idx * (N / 2) * (N / 2)];
    int* edge_start = &edges[idx * ((N / 2) * (N / 2) * 2) * 4];

    // Initialize UnionFindGPU
    UnionFindGPU uf((N / 2) * (N / 2), parent, rank);

    // Initialize maze to walls
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            maze[row * N + col] = (row % 2 == 0 || col % 2 == 0) ? MAZE_PATH::WALL : MAZE_PATH::EMPTY;
        }
    }

    int edgesCount = 0;

    // Initialize edges list
    for (int row = 1; row < N; row += 2) {
        for (int col = 1; col < N; col += 2) {
            if (row < N - 2) { // Vertical walls
                edge_start[edgesCount * 4 + 0] = row;
                edge_start[edgesCount * 4 + 1] = col;
                edge_start[edgesCount * 4 + 2] = row + 2;
                edge_start[edgesCount * 4 + 3] = col;
                edgesCount++;
            }
            if (col < N - 2) { // Horizontal walls
                edge_start[edgesCount * 4 + 0] = row;
                edge_start[edgesCount * 4 + 1] = col;
                edge_start[edgesCount * 4 + 2] = row;
                edge_start[edgesCount * 4 + 3] = col + 2;
                edgesCount++;
            }
        }
    }

    // Apply Kruskal's algorithm with random sampling without replacement
    for (int i = edgesCount - 1; i >= 0; --i) {
        // Randomly select an edge from the remaining edges
        int randIdx = curand(&localState) % (i + 1);

        // Access the selected edge
        int row1 = edge_start[randIdx * 4 + 0];
        int col1 = edge_start[randIdx * 4 + 1];
        int row2 = edge_start[randIdx * 4 + 2];
        int col2 = edge_start[randIdx * 4 + 3];

        int cell1 = (row1 / 2) * (N / 2) + (col1 / 2);
        int cell2 = (row2 / 2) * (N / 2) + (col2 / 2);

        // If the two cells are not already connected
        if (uf.union_sets(cell1, cell2)) {
            // Remove the wall between the two cells
            maze[(row1 + row2) / 2 * N + (col1 + col2) / 2] = MAZE_PATH::EMPTY;
        }

        // Swap the selected edge with the last edge in the list
        for (int k = 0; k < 4; ++k) {
            int temp = edge_start[i * 4 + k];
            edge_start[i * 4 + k] = edge_start[randIdx * 4 + k];
            edge_start[randIdx * 4 + k] = temp;
        }
    }

    globalState[idx] = localState;
}



int parallel_kruskal_with_sequential_combine() {
    auto start = std::chrono::high_resolution_clock::now();

    int num_mazes = P * P;
    int maze_size = N * N;
    int large_size = LARGE_MAZE_SIZE * LARGE_MAZE_SIZE;

    // Allocate memory for the mazes on the GPU
    MAZE_PATH* d_mazes;
    cudaMalloc(&d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH));

    // Allocate memory for RNG states on the GPU
    curandState* d_states;
    cudaMalloc(&d_states, num_mazes * sizeof(curandState));

    // Initialize the RNG states
    init_rng << <P, P >> > (d_states, static_cast<unsigned long>(time(NULL)));
    cudaCheckError();

    cudaDeviceSynchronize();

    // Generate the mazes using Kruskal's algorithm
    int* d_parentArray;
    cudaMalloc(&d_parentArray, N/2 * N/2 * num_mazes * sizeof(int));
    int* d_rankArray;
    cudaMalloc(&d_rankArray, N/2 * N/2 * num_mazes * sizeof(int));
    // Allocate memory for edges on the device
    int maxEdges = N/2 * N/2 * 2; // Adjust according to your logic
    int* d_edges;
    cudaMalloc(&d_edges, num_mazes * maxEdges * 4 * sizeof(int));

    // Now you can use d_edges directly in your kernel
    generate_mazes_kruskal << <P, P >> > (d_states, d_mazes, d_parentArray, d_rankArray, d_edges);
    cudaCheckError();

    cudaFree(d_parentArray);
    cudaFree(d_rankArray);
    cudaFree(d_edges);

    cudaDeviceSynchronize();

    auto endGeneration = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedGeneration = endGeneration - start;

    std::cout << "Maze generation took " << elapsedGeneration.count() << " ms" << std::endl;

    // Copy the mazes back to the host
    MAZE_PATH* h_mazes = (MAZE_PATH*)malloc(maze_size * num_mazes * sizeof(MAZE_PATH));
    cudaMemcpy(h_mazes, d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH), cudaMemcpyDeviceToHost);
    cudaCheckError();

    auto endDataTransfer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedDataTransfer = endDataTransfer - endGeneration;
    std::cout << "Data transfer before combining took " << elapsedDataTransfer.count() << " ms" << std::endl;

    // Combine the mazes on the CPU
    MAZE_PATH* h_large_maze = (MAZE_PATH*)malloc(large_size * sizeof(MAZE_PATH));
    combine_mazes_cpu(h_mazes, h_large_maze);

    auto endCombine = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedCombine = endCombine - endDataTransfer;
    std::cout << "Combine_mazes took " << elapsedCombine.count() << " ms" << std::endl;

    // Use Union-Find to connect the mazes on the CPU
    UnionFind uf(num_mazes);
    connect_mazes(h_large_maze, uf);

    auto endConnect = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedConnect = endConnect - endCombine;
    std::cout << "Maze connection took " << elapsedConnect.count() << " ms" << std::endl;

    std::chrono::duration <double, std::milli> totalElapsed = endConnect - start;
    std::cout << "Total time: " << totalElapsed.count() << " ms" << std::endl;

    // Print the combined large maze
    //print_combined_maze(h_large_maze, LARGE_MAZE_SIZE);

    // Free resources
    cudaFree(d_mazes);
    cudaFree(d_states);
    cudaFree(d_parentArray);
    cudaFree(d_rankArray);
    free(h_mazes);
    free(h_large_maze);

    return 0;
}

int parallel_kruskal_with_parallel_combine() {
    auto start = std::chrono::high_resolution_clock::now();

    int num_mazes = P * P;
    int maze_size = N * N;
    int large_size = LARGE_MAZE_SIZE * LARGE_MAZE_SIZE;

    // Allocate memory for the mazes on the GPU
    MAZE_PATH* d_mazes;
    cudaMalloc(&d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH));

    // Allocate memory for RNG states on the GPU
    curandState* d_states;
    cudaMalloc(&d_states, num_mazes * sizeof(curandState));

    // Initialize the RNG states
    init_rng << <P, P >> > (d_states, static_cast<unsigned long>(time(NULL)));
    cudaCheckError();

    cudaDeviceSynchronize();

    // Generate the mazes using Kruskal's algorithm
    int* d_parentArray;
    cudaMalloc(&d_parentArray, N / 2 * N / 2 * num_mazes * sizeof(int));
    int* d_rankArray;
    cudaMalloc(&d_rankArray, N / 2 * N / 2 * num_mazes * sizeof(int));
    // Allocate memory for edges on the device
    int maxEdges = N / 2 * N / 2 * 2; // Adjust according to your logic
    int* d_edges;
    cudaMalloc(&d_edges, num_mazes * maxEdges * 4 * sizeof(int));


    // Now you can use d_edges directly in your kernel
    generate_mazes_kruskal << <P, P >> > (d_states, d_mazes, d_parentArray, d_rankArray, d_edges);
    cudaDeviceSynchronize();
    cudaCheckError();


    cudaFree(d_parentArray);
    cudaFree(d_rankArray);
    cudaFree(d_edges);

    auto endGeneration = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedGeneration = endGeneration - start;

    std::cout << "Maze generation took " << elapsedGeneration.count() << " ms" << std::endl;

    // Allocate memory for the combined large maze on the GPU
    MAZE_PATH* d_large_maze;
    cudaMalloc(&d_large_maze, large_size * sizeof(MAZE_PATH));

    // Define block and grid dimensions for combining mazes
    int blockSize = 16;  // 32x32 block size
    dim3 blockDim(blockSize, blockSize, 1);
    int gridX = (N + blockSize - 1) / blockSize;
    int gridY = (N + blockSize - 1) / blockSize;
    dim3 gridDim(gridX, gridY, P * P);

    // Combine the mazes on the GPU
    combine_mazes << <gridDim, blockDim >> > (d_mazes, d_large_maze, N, LARGE_MAZE_SIZE, P);
    cudaCheckError();

    cudaDeviceSynchronize();

    auto endCombine = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedCombine = endCombine - endGeneration;

    std::cout << "Combine_mazes took " << elapsedCombine.count() << " ms" << std::endl;

    // Copy the combined large maze back to the host
    MAZE_PATH* h_large_maze = (MAZE_PATH*)malloc(large_size * sizeof(MAZE_PATH));
    cudaMemcpy(h_large_maze, d_large_maze, large_size * sizeof(MAZE_PATH), cudaMemcpyDeviceToHost);
    cudaCheckError();

    auto endDataTransfer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedDataTransfer = endDataTransfer - endCombine;
    std::cout << "Data transfer before connecting took " << elapsedDataTransfer.count() << " ms" << std::endl;

    // Use Union-Find to connect the mazes on the CPU

    UnionFind uf(num_mazes);
    connect_mazes(h_large_maze, uf);

    auto endConnect = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedConnect = endConnect - endDataTransfer;
    std::cout << "Maze connection took " << elapsedConnect.count() << " ms" << std::endl;

    std::chrono::duration <double, std::milli> totalElapsed = endConnect - start;
    std::cout << "Total time: " << totalElapsed.count() << " ms" << std::endl;

    // Print the combined large maze
    //print_combined_maze(h_large_maze, LARGE_MAZE_SIZE);

    // Free resources
    cudaFree(d_mazes);
    cudaFree(d_states);
    cudaFree(d_large_maze);
    free(h_large_maze);

    return 0;
}

int seq_DFS_single_maze() {
    int maze_size = 18181;
    //int maze_size = 63;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<MAZE_PATH> maze;
    int exit_row, exit_col;

    std::random_device rd;
    std::mt19937 gen(rd());

    initialize_maze_cpu(maze, maze_size, exit_row, exit_col, gen);
    dfs_maze_generation_cpu(maze, maze_size, exit_row, exit_col, gen);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Sequential single maze generation took " << elapsed.count() << " ms" << std::endl;

    //print_combined_maze(maze.data(), maze_size);

    return 0;

}


int parallel_DFS_with_parallel_combine() {
    auto start = std::chrono::high_resolution_clock::now();

    int num_mazes = P * P;
    int maze_size = N * N;
    int large_size = LARGE_MAZE_SIZE * LARGE_MAZE_SIZE;

    // Allocate memory for the mazes on the GPU
    MAZE_PATH* d_mazes;
    cudaMalloc(&d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH));

    // Allocate memory for RNG states on the GPU
    curandState* d_states;
    cudaMalloc(&d_states, num_mazes * sizeof(curandState));

    // Initialize the RNG states
    init_rng << <P, P >> > (d_states, static_cast<unsigned long>(time(NULL)));
    cudaCheckError();

    cudaDeviceSynchronize();

    // Generate the mazes
    generate_mazes << <P, P >> > (d_states, d_mazes);
    cudaCheckError();

    cudaDeviceSynchronize();

    auto endGeneration = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedGeneration = endGeneration - start;

    std::cout << "Maze generation took " << elapsedGeneration.count() << " ms" << std::endl;

    // Allocate memory for the combined large maze on the GPU
    MAZE_PATH* d_large_maze;
    cudaMalloc(&d_large_maze, large_size * sizeof(MAZE_PATH));

    int blockSize = 16;  // 32x32 block size
    dim3 blockDim(blockSize, blockSize, 1);
    int gridX = (N + blockSize - 1) / blockSize;
    int gridY = (N + blockSize - 1) / blockSize;
    dim3 gridDim(gridX, gridY, P * P);

    // Combine the mazes on the GPU
    combine_mazes << <gridDim, blockDim >> > (d_mazes, d_large_maze, N, LARGE_MAZE_SIZE, P);
    cudaCheckError();

    cudaDeviceSynchronize();

    auto endCombine = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedCombine = endCombine - endGeneration;

    std::cout << "Combine_mazes took " << elapsedCombine.count() << " ms" << std::endl;


    // Copy the combined large maze back to the host
    MAZE_PATH* h_large_maze = (MAZE_PATH*)malloc(large_size * sizeof(MAZE_PATH));
    cudaMemcpy(h_large_maze, d_large_maze, large_size * sizeof(MAZE_PATH), cudaMemcpyDeviceToHost);
    cudaCheckError();

    auto endDataTransfer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedDataTransfer = endDataTransfer - endCombine;

    std::cout << "Data transfer before connecting took " << elapsedDataTransfer.count() << " ms" << std::endl;

    // Use Union-Find to connect the mazes on the CPU
    UnionFind uf(num_mazes);
    connect_mazes(h_large_maze, uf);


    auto endConnect = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedConnect = endConnect - endDataTransfer;

    std::cout << "Maze connection took " << elapsedConnect.count() << " ms" << std::endl;

    std::chrono::duration <double, std::milli> totalElapsed = endConnect - start;
    std::cout << "Total time: " << totalElapsed.count() << " ms" << std::endl;
    // Print the combined large maze
    //print_combined_maze(h_large_maze, LARGE_MAZE_SIZE);

    // Free resources
    cudaFree(d_mazes);
    cudaFree(d_states);
    cudaFree(d_large_maze);
    free(h_large_maze);

    return 0;
}


// sequential combination of mazes
int parallel_DFS_with_sequential_combine() {
    auto start = std::chrono::high_resolution_clock::now();

    int num_mazes = P * P;
    int maze_size = N * N;

    // Allocate memory for the mazes on the GPU
    MAZE_PATH* d_mazes;
    cudaMalloc(&d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH));

    // Allocate memory for RNG states on the GPU
    curandState* d_states;
    cudaMalloc(&d_states, num_mazes * sizeof(curandState));

    // Initialize the RNG states
    init_rng << <P, P >> > (d_states, static_cast<unsigned long>(time(NULL)));
    cudaCheckError();

    cudaDeviceSynchronize();

    // Generate the mazes
    generate_mazes << <P, P >> > (d_states, d_mazes);
    cudaCheckError();

    cudaDeviceSynchronize();

    auto endGeneration = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedGeneration = endGeneration - start;

    std::cout << "Maze generation took " << elapsedGeneration.count() << " ms" << std::endl;

    // Allocate memory for the small mazes on the CPU
    MAZE_PATH* h_mazes = (MAZE_PATH*)malloc(maze_size * num_mazes * sizeof(MAZE_PATH));

    // Copy the small mazes from GPU to CPU
    cudaMemcpy(h_mazes, d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Allocate memory for the combined large maze
    MAZE_PATH* h_large_maze = (MAZE_PATH*)malloc(LARGE_MAZE_SIZE * LARGE_MAZE_SIZE * sizeof(MAZE_PATH));

    auto endDataTransfer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedDataTransfer = endDataTransfer - endGeneration;

    std::cout << "Data transfer before maze combine took " << elapsedDataTransfer.count() << " ms" << std::endl;

    // Combine the small mazes into a large maze on the CPU
    combine_mazes_cpu(h_mazes, h_large_maze);

    auto endCombine = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = endCombine - endDataTransfer;

    std::cout << "Combine_mazes_cpu took " << elapsed.count() << " ms" << std::endl;

    // Use Union-Find to connect the mazes on the CPU
    UnionFind uf(num_mazes);
    connect_mazes(h_large_maze, uf);

    auto endConnect = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedConnect = endConnect - endCombine;

    std::cout << "Maze connection took " << elapsedConnect.count() << " ms" << std::endl;

    std::chrono::duration <double, std::milli> totalElapsed = endConnect - start;
    std::cout << "Total time: " << totalElapsed.count() << " ms" << std::endl;

    // Print the combined large maze
    //print_combined_maze(h_large_maze, LARGE_MAZE_SIZE);

    // Free resources
    cudaFree(d_mazes);
    cudaFree(d_states);
    free(h_mazes);
    free(h_large_maze);

    return 0;
}

// Kruskal's algorithm for maze generation on the CPU
void kruskal_maze_generation_cpu(std::vector<MAZE_PATH>& maze, int size, std::mt19937& rng) {
    UnionFind uf((size / 2) * (size / 2));

    // List of all edges (walls) between cells
    std::vector<std::tuple<int, int, int, int>> edges;

    for (int row = 1; row < size; row += 2) {
        for (int col = 1; col < size; col += 2) {
            if (row < size - 2) { // Vertical walls
                edges.emplace_back(row, col, row + 2, col);
            }
            if (col < size - 2) { // Horizontal walls
                edges.emplace_back(row, col, row, col + 2);
            }
        }
    }

    // Shuffle the edges randomly
    std::shuffle(edges.begin(), edges.end(), rng);

    for (const auto& edge : edges) {
        int row1 = std::get<0>(edge);
        int col1 = std::get<1>(edge);
        int row2 = std::get<2>(edge);
        int col2 = std::get<3>(edge);

        int cell1 = (row1 / 2) * (size / 2) + (col1 / 2);
        int cell2 = (row2 / 2) * (size / 2) + (col2 / 2);

        // If the two cells are not already connected
        if (uf.union_sets(cell1, cell2)) {
            // Remove the wall between the two cells
            maze[(row1 + row2) / 2 * size + (col1 + col2) / 2] = MAZE_PATH::EMPTY;
        }
    }
}


int seq_Kruskal_single_maze() {
    int maze_size = 18181;
    //int maze_size = 63;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<MAZE_PATH> maze;
    int exit_row, exit_col;

    std::random_device rd;
    std::mt19937 gen(rd());

    initialize_maze_cpu(maze, maze_size, exit_row, exit_col, gen);
    kruskal_maze_generation_cpu(maze, maze_size, gen);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Sequential single maze generation took " << elapsed.count() << " ms" << std::endl;

    //print_combined_maze(maze.data(), maze_size);

    return 0;
}



int main() {
    // Get device properties

    // Max threads per block: 1024
    // Max threads per multiprocessor : 2048
    // Number of multiprocessors : 3

    /*
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    */

    std::cout << "________Running parallel DFS with parallel combine________" << std::endl << std::endl;
    parallel_DFS_with_parallel_combine();

    std::cout << std::endl << std::endl << "________Running parallel DFS with sequential combine________" << std::endl << std::endl;
    parallel_DFS_with_sequential_combine();

    std::cout << std::endl << std::endl << "________Running sequential DFS single maze generation________" << std::endl << std::endl;
    seq_DFS_single_maze();


    std::cout << std::endl << std::endl << "________Running parallel Kruskal maze with parallel combine________" << std::endl << std::endl;
    parallel_kruskal_with_parallel_combine();

    std::cout << std::endl << std::endl << "________Running parallel Kruskal maze with sequential combine________" << std::endl << std::endl;
    parallel_kruskal_with_sequential_combine();

    std::cout << std::endl << std::endl << "________Running sequential Kruskal single maze generation________" << std::endl << std::endl;
    seq_Kruskal_single_maze();

    return 0;
}