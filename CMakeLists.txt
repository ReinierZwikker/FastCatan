cmake_minimum_required(VERSION 3.22)

project(FastCatan)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define source files
set(SOURCES
        main.cpp
        src/game/board.cpp
        src/game/board.h
        src/game/components.h
        src/game/game.cpp
        src/game/game.h
        src/game/player.cpp
        src/game/player.h
        src/game/HumanPlayer/human_player.cpp
        src/game/HumanPlayer/human_player.h
        app/app.cpp
        app/viewport.cpp
        app/window_board.cpp
        app/3rd_party/imgui/imgui.cpp
        app/3rd_party/imgui/imgui_demo.cpp
        app/3rd_party/imgui/imgui_draw.cpp
        app/3rd_party/imgui/imgui_tables.cpp
        app/3rd_party/imgui/imgui_widgets.cpp
        app/3rd_party/imgui/backends/imgui_impl_win32.cpp
        app/3rd_party/imgui/backends/imgui_impl_opengl3.cpp
)

set(TEST_SOURCES
        app/test.cpp
        app/viewport.cpp
        app/3rd_party/imgui/imgui.cpp
        app/3rd_party/imgui/imgui_demo.cpp
        app/3rd_party/imgui/imgui_draw.cpp
        app/3rd_party/imgui/imgui_tables.cpp
        app/3rd_party/imgui/imgui_widgets.cpp
        app/3rd_party/imgui/backends/imgui_impl_win32.cpp
        app/3rd_party/imgui/backends/imgui_impl_opengl3.cpp
)

# Create executable
add_executable(FastCatan ${SOURCES})
add_executable(Test ${TEST_SOURCES})

# Include directories
target_include_directories(FastCatan PRIVATE
        src
        app/3rd_party/imgui
        app/3rd_party/imgui/backends
        app)
target_include_directories(Test PRIVATE
        src
        app/3rd_party/imgui
        app/3rd_party/imgui/backends
        app)


# Find OpenGL package
find_package(OpenGL REQUIRED)

# Link against necessary libraries
target_link_libraries(FastCatan PRIVATE
        OpenGL::GL
        Dwmapi.lib
)
target_link_libraries(Test PRIVATE
        OpenGL::GL
        Dwmapi.lib
)

# Link against additional libraries for Windows
if(WIN32)
    target_link_libraries(FastCatan PRIVATE
            opengl32
            gdi32
            user32
    )
    target_link_libraries(Test PRIVATE
            opengl32
            gdi32
            user32
    )
endif()
