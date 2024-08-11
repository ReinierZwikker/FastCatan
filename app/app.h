#ifndef FASTCATAN_APP_H
#define FASTCATAN_APP_H

#include "../app/3rd_party/imgui/imgui.h"
#include "../app/3rd_party/imgui/backends/imgui_impl_opengl3.h"
#include "../app/3rd_party/imgui/backends/imgui_impl_win32.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <GL/GL.h>
#include <tchar.h>

// Include other GUI
#include "viewport.h"
#include "window_board.h"
#include "window_game.h"
#include "window_player.h"
#include "window_ai.h"
#include "window_replay.h"


class App {
  public:
    App(int, char**, Game*);
    ~App();
    void Refresh();

    bool done = false;

  private:
    // Windows
    ViewPort viewport = ViewPort();  // Catan itself

    // Show Bools
    bool show_demo_window = false;
    bool show_board_window = false;
    bool show_game_window = false;
    bool show_player_window[4] = {true, false, false, false};
    bool show_ai_window = true;
    bool show_replay_window = true;

    bool training_in_progress = false;
    bool replay_in_progress = false;

    Game* game_pointer;
    WindowAI window_ai;
    WindowReplay window_replay;

    // TODO: Put into selection screen
    int amount_of_players = 4;

    // Data stored per platform window
    struct WGL_WindowData { HDC hDC; };

    // Data
    static HGLRC            g_hRC;
    static WGL_WindowData   g_MainWindow;
    static int              g_Width;
    static int              g_Height;

    // Forward declarations of helper functions
    static bool CreateDeviceWGL(HWND hWnd, WGL_WindowData* data);
    static void CleanupDeviceWGL(HWND hWnd, WGL_WindowData* data);
    static void ResetDeviceWGL();
    static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    // Initialize parameters for later use
    MSG msg{};
    ImGuiIO& io;
    HWND hwnd{};
    WNDCLASSEXW wc{};

    // State
    ImVec4 clear_color = ImVec4(0.0f, 0.6f, 0.7f, 1.00f);

    // Support functions for multi-viewports
    static void Hook_Renderer_CreateWindow(ImGuiViewport* viewport);
    static void Hook_Renderer_DestroyWindow(ImGuiViewport* viewport);
    static void Hook_Platform_RenderWindow(ImGuiViewport* viewport, void*);
    static void Hook_Renderer_SwapBuffers(ImGuiViewport* viewport, void*);
};

#endif //FASTCATAN_APP_H
