cmake_minimum_required(VERSION 3.22)

project(FastCatan)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define source files
set(MAIN_SOURCES
        main.cpp
)

set(GAME_SOURCES
        src/game/board.cpp
        src/game/components.h
        src/game/game.cpp
        src/game/game_manager.cpp
        src/game/player.cpp
        src/game/HumanPlayer/console_player.cpp
        src/game/HumanPlayer/gui_player.cpp
        src/game/AIHelpers/ai_helper.cpp
        src/game/AIHelpers/zwik_helper.cpp
        src/game/AIHelpers/bean_helper.cpp
        #        src/game/AIPlayer/ai_zwik_player.cpp
        src/game/AIPlayer/ai_bean_player.cpp
        src/game/AIPlayer/random_player.cpp
)

set(APP_SOURCES
        app/app.cpp
        app/viewport.cpp
        app/app_components.h
        app/windows/window.cpp
        app/windows/window_board.cpp
        app/windows/window_game.cpp
        app/windows/window_player.cpp
        app/windows/window_ai.cpp
        app/windows/window_replay.cpp
        app/3rd_party/imgui/imgui.cpp
        app/3rd_party/imgui/imgui_demo.cpp
        app/3rd_party/imgui/imgui_draw.cpp
        app/3rd_party/imgui/imgui_tables.cpp
        app/3rd_party/imgui/imgui_widgets.cpp
        app/3rd_party/imgui/backends/imgui_impl_win32.cpp
        app/3rd_party/imgui/backends/imgui_impl_opengl3.cpp
)

# Create executable
add_executable(FastCatan)
target_sources(FastCatan PRIVATE ${MAIN_SOURCES} ${APP_SOURCES} ${GAME_SOURCES})

# Custom Icon
set(IMAGE_SOURCE "${CMAKE_SOURCE_DIR}/app/logo.ico")
set(IMAGE_DESTINATION "${CMAKE_BINARY_DIR}/logo.ico")

add_custom_command(TARGET FastCatan
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${IMAGE_SOURCE} ${IMAGE_DESTINATION}
        COMMENT "Copying the icon to the build directory")


# Include directories
target_include_directories(FastCatan PRIVATE
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

# Link against additional libraries for Windows
if(WIN32)
    target_link_libraries(FastCatan PRIVATE
            opengl32
            gdi32
            user32
    )
endif()
