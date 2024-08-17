// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef RANDOMMAZESOLVER_UTILS_H
#define RANDOMMAZESOLVER_UTILS_H

#include "../external/json.hpp"

#include "../Sequential/maze/maze_generation.h"


nlohmann::json parse_configuration(const std::filesystem::path& project_folder);
std::filesystem::path find_project_path();
void display_ascii_maze(std::vector<std::vector<MAZE_PATH>>& maze, int& size, bool show_steps);
std::string generate_ascii_maze(std::vector<std::vector<MAZE_PATH>>& maze, int& size);

#endif //RANDOMMAZESOLVER_UTILS_H