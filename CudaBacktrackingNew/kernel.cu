﻿#include <curand_kernel.h>
#include <iostream>
#include <random>

#define N 21  // Size of individual mazes (N x N)
#define P 4  // Number of mazes in one row/column of the large maze
#define MAX_SIZE (N * N)
#define LARGE_SIZE (N * P)  // Size of the large maze (N*P x N*P)

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
        printf("curand state initialized for idx = %d\n", idx);
        // Test random number generation
        float random_number = curand_uniform(&state[idx]);
        printf("Random number generated by thread %d: %f\n", idx, random_number);
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

        bool is_exit = (maze[curr_row * size + curr_col] == MAZE_PATH::EXIT);
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
    int maze_size = N * N;
    // Iterate over each maze block
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            // Copy the individual maze into the correct position in the large maze
            for (int row = 0; row < N; ++row) {
                for (int col = 0; col < N; ++col)
                    large_maze[(i * N + row) * LARGE_SIZE + (j * N + col)] =
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
    int maze_size = N * N;

    std::vector<std::pair<int, int>> directions = {
        {0, -1}, {0, 1}, {-1, 0}, {1, 0}
    };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> maze_dist(0, total_mazes - 1);
    std::uniform_int_distribution<> dir_dist(0, directions.size() - 1);

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
                large_maze[wall_row_a * LARGE_SIZE + wall_col_a] = MAZE_PATH::EMPTY;
                large_maze[wall_row_b * LARGE_SIZE + wall_col_b] = MAZE_PATH::EMPTY;
            }
        }
    }
}



int parallel_combine() {
    int num_mazes = P * P;
    int maze_size = N * N;
    int large_size = LARGE_SIZE * LARGE_SIZE;

    // Allocate memory for the mazes on the GPU
    MAZE_PATH* d_mazes;
    cudaMalloc(&d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH));

    // Allocate memory for RNG states on the GPU
    curandState* d_states;
    cudaMalloc(&d_states, num_mazes * sizeof(curandState));

    // Initialize the RNG states
    init_rng << <P, P >> > (d_states, time(NULL));
    cudaCheckError();

    cudaDeviceSynchronize();

    // Generate the mazes
    generate_mazes << <P, P >> > (d_states, d_mazes);
    cudaCheckError();

    cudaDeviceSynchronize();

    // Allocate memory for the combined large maze on the GPU
    MAZE_PATH* d_large_maze;
    cudaMalloc(&d_large_maze, large_size * sizeof(MAZE_PATH));

    // Define block and grid dimensions for combining mazes
    dim3 blockDim(N, N, 1);
    dim3 gridDim(P, P, num_mazes);

    // Combine the mazes on the GPU
    combine_mazes << <gridDim, blockDim >> > (d_mazes, d_large_maze, N, LARGE_SIZE, P);
    cudaCheckError();

    cudaDeviceSynchronize();

    // Copy the combined large maze back to the host
    MAZE_PATH* h_large_maze = (MAZE_PATH*)malloc(large_size * sizeof(MAZE_PATH));
    cudaMemcpy(h_large_maze, d_large_maze, large_size * sizeof(MAZE_PATH), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Use Union-Find to connect the mazes on the CPU
    UnionFind uf(num_mazes);
    connect_mazes(h_large_maze, uf);

    // Print the combined large maze
    print_combined_maze(h_large_maze, LARGE_SIZE);

    // Free resources
    cudaFree(d_mazes);
    cudaFree(d_states);
    cudaFree(d_large_maze);
    free(h_large_maze);

    return 0;
}

// sequential combination of mazes
int seq_combine() {
    int num_mazes = P * P;
    int maze_size = N * N;
    int large_size = LARGE_SIZE * LARGE_SIZE;

    // Allocate memory for the mazes on the GPU
    MAZE_PATH* d_mazes;
    cudaMalloc(&d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH));

    // Allocate memory for RNG states on the GPU
    curandState* d_states;
    cudaMalloc(&d_states, num_mazes * sizeof(curandState));

    // Initialize the RNG states
    init_rng << <P, P >> > (d_states, time(NULL));
    cudaCheckError();

    cudaDeviceSynchronize();

    // Generate the mazes
    generate_mazes << <P, P >> > (d_states, d_mazes);
    cudaCheckError();

    cudaDeviceSynchronize();

    // Allocate memory for the small mazes on the CPU
    MAZE_PATH* h_mazes = (MAZE_PATH*)malloc(maze_size * num_mazes * sizeof(MAZE_PATH));

    // Copy the small mazes from GPU to CPU
    cudaMemcpy(h_mazes, d_mazes, maze_size * num_mazes * sizeof(MAZE_PATH), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Allocate memory for the combined large maze
    MAZE_PATH* h_large_maze = (MAZE_PATH*)malloc(LARGE_SIZE * LARGE_SIZE * sizeof(MAZE_PATH));

    combine_mazes_cpu(h_mazes, h_large_maze);

    // Print the combined large maze
    print_combined_maze(h_large_maze, LARGE_SIZE);

    // Free resources
    cudaFree(d_mazes);
    cudaFree(d_states);
    free(h_large_maze);

    return 0;
}

int main() {
    parallel_combine();
    //seq_combine();
    return 0;
}