// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <fstream>
#include <iostream>
#include <thread>

#include "../utils/utils.h"

// FUNCTIONS

/**
 * This function allows to parse and load the project's json configuration.
 *
 * All the items in the configuration are loaded into a proper structure.
 * @param project_folder This is the path related to the current project.
 * @return A json object that contains all the specified parameters.
 */
nlohmann::json parse_configuration(const std::filesystem::path& project_folder) {
    std::filesystem::path config_path = project_folder / "config" / "default.json";
    std::ifstream config_file;
    config_file.open(config_path);
    nlohmann::json json_config = nlohmann::json::parse(config_file);
    config_file.close();
    return json_config;
}

/**
 * This function allows to retrieve the current project's path in the filesystem.
 *
 * It perform a directory traversal until the current project's name folder is found.
 * The folder might be named as specified for this to work.
 * @return A path object representing the projects location.
 */
std::filesystem::path find_project_path() {
    std::filesystem::path project_folder = std::filesystem::current_path();
    while (!project_folder.string().ends_with("MazeGenerationGit"))
        if (project_folder.string() == "/") {
            project_folder.clear();
            break;
        }
        else {
            project_folder = project_folder.parent_path();
        }
    return project_folder;
}


/**
 * Prints the maze's inner structure by using ascii characters.
 *
 *  Keep in mind that the display's size is limited unfortunately. Big mazes will be hardly visible.
 *  @param maze It's the matrix representing the maze in it's current state.
 *  @param size Represents the length of each maze's side.
 *  @param show_steps Flag used to determine if each change step is being shown on screen.
 */
void display_ascii_maze(std::vector<std::vector<MAZE_PATH>>& maze, int& size, bool show_steps) {
    if (show_steps) {
        std::cout << "\x1B[2J\x1B[H";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::string ascii_maze = generate_ascii_maze(maze, size);
    std::cout << ascii_maze << std::endl << std::endl;
}


/**
 * Generates a string that represents the maze's inner structure by using ascii characters.
 *
 * This function simply reads the content of the maze's matrix and appends it's corresponding characters
 * to a string in order to visualize it.
 *
 *  @param maze It's the matrix representing the maze in it's current state.
 *  @param size Represents the length of each maze's side.
 */
std::string generate_ascii_maze(std::vector<std::vector<MAZE_PATH>>& maze, int& size) {

    const std::string RESET_COLOR = "\033[0m";
    const std::string SOLUTION_COLOR = "\033[31m"; // Red color for solution

    std::string ascii_maze;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            MAZE_PATH curr_path = maze[row][col];
            if (curr_path == MAZE_PATH::EMPTY)
                ascii_maze += "   ";
            else if (curr_path == MAZE_PATH::EXIT)
                ascii_maze += "  E";
            else if (curr_path == MAZE_PATH::WALL)
                ascii_maze += "  #";
            else if (curr_path == MAZE_PATH::START)
                ascii_maze += "  S";
            else if (curr_path == MAZE_PATH::SOLUTION)
                ascii_maze += "  x";
            //ascii_maze += SOLUTION_COLOR + "  x" + RESET_COLOR;
            else if (curr_path == MAZE_PATH::PARTICLE)
                ascii_maze += "  o";
        }
        ascii_maze += "\n";
    }

    return ascii_maze;
}