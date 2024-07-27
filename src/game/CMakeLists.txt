cmake_minimum_required(VERSION 3.22)
project(FastCatan)

set(CMAKE_CXX_STANDARD 23)

add_executable(FastCatan
        game.cpp
        game.h
        player.cpp
        player.h
        board.cpp
        board.h
        components.h)
