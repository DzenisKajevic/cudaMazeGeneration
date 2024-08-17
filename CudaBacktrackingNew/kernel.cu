#include <curand_kernel.h>
#include <iostream>

#define N 9  // Size of individual mazes (N x N)
#define P 2  // Number of mazes in one row/column of the large maze
#define MAX_SIZE (N * N)

// (N*P)*(N*P)

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

// Declaration (if defined later or in another file)
__device__ void initialize_maze_cuda(MAZE_PATH* maze, int size, int* exit_row, int* exit_col, curandState* localState);
__global__ void init_rng(curandState* state, unsigned long seed);
__global__ void generate_mazes(curandState* globalState, MAZE_PATH* mazes);
__device__ void generate_paths_cuda(MAZE_PATH* maze, int size, int* exit_coords, curandState* localState);
__device__ void visit_forward_cuda(MAZE_PATH* maze, int size, int& curr_index, int* curr_cell, int* curr_track, bool* visited_cells, curandState* localState, bool is_exit);
__device__ void get_unvisited_near_cells_cuda(MAZE_PATH* maze, int* curr_cell, int size, bool* visited_cells, int& n_cells, bool is_exit, int* near_cells);
__device__ void backtrack_cuda(MAZE_PATH* maze, int size, int& curr_index, int* curr_cell, int* curr_track, bool* visited_cells, curandState* localState);

void combine_mazes(MAZE_PATH* mazes, int n, int p, MAZE_PATH* large_maze);
void print_maze(MAZE_PATH* maze, int size);

// Function to print the maze
void print_maze(MAZE_PATH* maze, int size) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            switch (maze[row * size + col]) {
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
        }
        std::cout << std::endl;
    }
}

void print_maze_debug(MAZE_PATH* maze, int size) {
    for (int i = 0; i < size * size; ++i) {
        std::cout << static_cast<int>(maze[i]) << " ";
        if ((i + 1) % size == 0) std::cout << std::endl;
    }
}



// Kernel function to initialize random states
__global__ void init_rng(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// GPU function to initialize the maze (similar to initialize_maze)
__device__ void initialize_maze_cuda(MAZE_PATH* maze, int size, int* exit_row, int* exit_col, curandState* localState) {
    for (int i = 0; i < size * size; ++i) {
        int row = i / size;
        int col = i % size;
        if (row % 2 == 0 || col % 2 == 0) {
            maze[i] = MAZE_PATH::WALL;
        }
        else {
            maze[i] = MAZE_PATH::EMPTY;
        }
    }

    // Randomly choose a border for the exit (0: top/bottom, 1: left/right)
    int border_choice = curand(localState) % 2;

    //int exit_row, exit_col;

    if (border_choice == 0) {
        // Top or bottom border
        *exit_row = (curand(localState) % 2 == 0) ? 0 : size - 1;
        *exit_col = (curand(localState) % (size / 2)) * 2 + 1;
    }
    else {
        // Left or right border
        *exit_row = (curand(localState) % (size / 2)) * 2 + 1;
        *exit_col = (curand(localState) % 2 == 0) ? 0 : size - 1;
    }
    // Ensure the exit is within bounds (mozda bude trebalo)
    //if (exit_row < size && exit_col < size) {
    //    maze[exit_row * size + exit_col] = MAZE_PATH::EXIT;
    //}
    
    maze[*exit_row * size + *exit_col] = MAZE_PATH::EXIT;
}

// Kernel function to generate individual mazes
__global__ void generate_mazes(curandState* globalState, MAZE_PATH* mazes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get the random state for this thread
    curandState localState = globalState[idx];

    // Pointer to this thread's maze in the global memory
    MAZE_PATH* maze = &mazes[idx * N * N];

    int exit_row, exit_col;

    // Maze generation logic (pseudo-code)



    // Initialize the maze
    initialize_maze_cuda(maze, N, &exit_row, &exit_col, &localState);

    // Generate the paths
    generate_paths_cuda(maze, N, new int[2]{ exit_row, exit_col }, &localState);

    // Store the random state back to global memory
    globalState[idx] = localState;
}

// Function to combine mazes into one large maze on the CPU
void combine_mazes(MAZE_PATH* mazes, int n, int p, MAZE_PATH* large_maze) {
    int large_size = n * p;

    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int x = 0; x < n; ++x) {
                for (int y = 0; y < n; ++y) {
                    large_maze[(i * n + x) * large_size + (j * n + y)] = mazes[(i * p + j) * n * n + x * n + y];
                }
            }
        }
    }
}

__device__ void generate_paths_cuda(MAZE_PATH* maze, int size, int* exit_coords, curandState* localState) {
    bool* visited_cells = new bool[MAX_SIZE];  // Dynamically allocate
    int* curr_track = new int[MAX_SIZE * 2];   // Dynamically allocate
    int curr_index = 0;

    int curr_cell[2] = { exit_coords[0], exit_coords[1] };
    visited_cells[curr_cell[0] * size + curr_cell[1]] = true;
    curr_track[curr_index * 2] = curr_cell[0];
    curr_track[curr_index * 2 + 1] = curr_cell[1];

    visit_forward_cuda(maze, size, curr_index, curr_cell, curr_track, visited_cells, localState, true);

    delete[] visited_cells;
    delete[] curr_track;
}

__device__ void visit_forward_cuda(MAZE_PATH* maze, int size, int& curr_index, int* curr_cell, int* curr_track, bool* visited_cells, curandState* localState, bool is_exit) {
    int n_cells = 0;
    int near_cells[4 * 2];  // Up to 4 possible cells, each with x and y coordinates

    get_unvisited_near_cells_cuda(maze, curr_cell, size, visited_cells, n_cells, is_exit, near_cells);
    while (n_cells > 0) {
        curr_index++;
        int new_cell_index = curand(localState) % n_cells;
        int* new_cell = &near_cells[new_cell_index * 2];

        if (!is_exit) {
            int row_to_del = (new_cell[0] + curr_cell[0]) / 2;
            int col_to_del = (new_cell[1] + curr_cell[1]) / 2;

            maze[row_to_del * size + col_to_del] = MAZE_PATH::EMPTY;
        }
        else {
            is_exit = false;
        }

        curr_cell[0] = new_cell[0];
        curr_cell[1] = new_cell[1];
        visited_cells[curr_cell[0] * size + curr_cell[1]] = true;
        curr_track[curr_index * 2] = curr_cell[0];
        curr_track[curr_index * 2 + 1] = curr_cell[1];

        get_unvisited_near_cells_cuda(maze, curr_cell, size, visited_cells, n_cells, is_exit, near_cells);
    }
    backtrack_cuda(maze, size, curr_index, curr_cell, curr_track, visited_cells, localState);
}

__device__ void get_unvisited_near_cells_cuda(MAZE_PATH* maze, int* curr_cell, int size, bool* visited_cells, int& n_cells, bool is_exit, int* near_cells) {
    n_cells = 0;
    int index_offset = is_exit ? 1 : 2;

    // East cell
    int index = curr_cell[0] + index_offset;
    if (index < size && !visited_cells[index * size + curr_cell[1]]) {
        near_cells[n_cells * 2] = index;
        near_cells[n_cells * 2 + 1] = curr_cell[1];
        n_cells++;
    }

    // North cell
    index = curr_cell[1] + index_offset;
    if (index < size && !visited_cells[curr_cell[0] * size + index]) {
        near_cells[n_cells * 2] = curr_cell[0];
        near_cells[n_cells * 2 + 1] = index;
        n_cells++;
    }

    // West cell
    index = curr_cell[0] - index_offset;
    if (index >= 0 && !visited_cells[index * size + curr_cell[1]]) {
        near_cells[n_cells * 2] = index;
        near_cells[n_cells * 2 + 1] = curr_cell[1];
        n_cells++;
    }

    // South cell
    index = curr_cell[1] - index_offset;
    if (index >= 0 && !visited_cells[curr_cell[0] * size + index]) {
        near_cells[n_cells * 2] = curr_cell[0];
        near_cells[n_cells * 2 + 1] = index;
        n_cells++;
    }
}

__device__ void backtrack_cuda(MAZE_PATH* maze, int size, int& curr_index, int* curr_cell, int* curr_track, bool* visited_cells, curandState* localState) {
    int n_cells = 0;

    for (int index = curr_index - 1; index > 0; index--) {
        curr_cell[0] = curr_track[index * 2];
        curr_cell[1] = curr_track[index * 2 + 1];
        curr_index = index;

        get_unvisited_near_cells_cuda(maze, curr_cell, size, visited_cells, n_cells, false, curr_track);
        if (n_cells > 0) {
            visit_forward_cuda(maze, size, curr_index, curr_cell, curr_track, visited_cells, localState, false);
            break;
        }
    }
}



int main() {
    int num_mazes = P * P;
    int maze_size = N * N;
    size_t mazes_bytes = num_mazes * maze_size * sizeof(MAZE_PATH);

    // Allocate memory for the mazes on the GPU
    MAZE_PATH* d_mazes;
    cudaMalloc(&d_mazes, mazes_bytes);

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

    // Allocate memory for one maze on the CPU
    MAZE_PATH* h_single_maze = new MAZE_PATH[maze_size];

    // Copy and print each maze
    for (int i = 0; i < num_mazes; ++i) {
        // Copy the maze from the GPU to the CPU
        cudaMemcpy(h_single_maze, d_mazes + i * maze_size, maze_size * sizeof(MAZE_PATH), cudaMemcpyDeviceToHost);
        //cudaCheckError();

        // Print the maze
        print_maze(h_single_maze, N);

        //print_maze_debug(h_single_maze, N);

        // Optionally, add a separator between mazes
        std::cout << "--------------------\n";
    }

    //// Copy one maze from the GPU to the CPU (e.g., the first maze)
    //cudaMemcpy(h_single_maze, d_mazes, maze_size * sizeof(MAZE_PATH), cudaMemcpyDeviceToHost);
    //
    //// Print the maze
    //print_maze(h_single_maze, N);

    // Allocate memory for the large maze on the CPU
    MAZE_PATH* large_maze = new MAZE_PATH[N * P * N * P];

    // Combine the mazes into one large maze
    MAZE_PATH* h_mazes = new MAZE_PATH[maze_size * num_mazes];
    cudaMemcpy(h_mazes, d_mazes, mazes_bytes, cudaMemcpyDeviceToHost);
    combine_mazes(h_mazes, N, P, large_maze);


    // Free resources
    cudaFree(d_mazes);
    cudaFree(d_states);
    delete[] h_mazes;
    delete[] large_maze;

    return 0;
}