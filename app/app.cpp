// Dear ImGui: standalone example application for Win32 + OpenGL 3

// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp

// This is provided for completeness, however it is strongly recommended you use OpenGL with SDL or GLFW.

#include "app.h"

#include "../app/3rd_party/imgui/imgui.h"
#include "../app/3rd_party/imgui/backends/imgui_impl_opengl3.h"
#include "../app/3rd_party/imgui/backends/imgui_impl_win32.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <GL/GL.h>
#include <tchar.h>
#include <mutex>


// Load the icon and set it for the window
void SetWindowIcon(HWND hwnd) {
  HICON hIcon = (HICON)LoadImage(NULL, "logo.ico", IMAGE_ICON, 32, 32, LR_LOADFROMFILE);

  if (hIcon) {
    // Set the big icon (taskbar, window)
    SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);

    // Set the small icon (title bar)
    SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);
  } else {
    MessageBox(hwnd, "Failed to load icon!", "Error", MB_OK | MB_ICONERROR);
  }
}


// Support function for multi-viewports
// Unlike most other backend combination, we need specific hooks to combine Win32+OpenGL.
// We could in theory decide to support Win32-specific code in OpenGL backend via e.g. an hypothetical ImGui_ImplOpenGL3_InitForRawWin32().
void App::Hook_Renderer_CreateWindow(ImGuiViewport* viewport)
{
  assert(viewport->RendererUserData == NULL);

  WGL_WindowData* data = IM_NEW(WGL_WindowData);
  CreateDeviceWGL((HWND)viewport->PlatformHandle, data);
  viewport->RendererUserData = data;
}

void App::Hook_Renderer_DestroyWindow(ImGuiViewport* viewport)
{
  if (viewport->RendererUserData != NULL)
  {
    WGL_WindowData* data = (WGL_WindowData*)viewport->RendererUserData;
    CleanupDeviceWGL((HWND)viewport->PlatformHandle, data);
    IM_DELETE(data);
    viewport->RendererUserData = NULL;
  }
}

void App::Hook_Platform_RenderWindow(ImGuiViewport* viewport, void*)
{
  // Activate the platform window DC in the OpenGL rendering context
  if (WGL_WindowData* data = (WGL_WindowData*)viewport->RendererUserData)
    wglMakeCurrent(data->hDC, g_hRC);
}

void App::Hook_Renderer_SwapBuffers(ImGuiViewport* viewport, void*)
{
  if (WGL_WindowData* data = (WGL_WindowData*)viewport->RendererUserData)
    ::SwapBuffers(data->hDC);
}

// Helper function to get ImGuiIO reference
ImGuiIO& initializeImGuiIO() {
  if (ImGui::GetCurrentContext() == nullptr) {
    ImGui::CreateContext();
  }
  return ImGui::GetIO();
}

int App::g_Width = 800;       // Default width
int App::g_Height = 600;      // Default height
App::WGL_WindowData App::g_MainWindow = {nullptr};
HGLRC App::g_hRC = nullptr;

App::App(int, char**, Game* game) : io(initializeImGuiIO()) {
  // Create application window
  // ImGui_ImplWin32_EnableDpiAwareness();
  wc = { sizeof(wc), CS_OWNDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"ImGui Example", nullptr };
  ::RegisterClassExW(&wc);
  hwnd = ::CreateWindowW(wc.lpszClassName, L"Fast Catan", WS_OVERLAPPEDWINDOW, 100, 100, 1280, 800, nullptr, nullptr, wc.hInstance, nullptr);

  // Initialize OpenGL
  if (!CreateDeviceWGL(hwnd, &g_MainWindow))
  {
    CleanupDeviceWGL(hwnd, &g_MainWindow);
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
  }
  wglMakeCurrent(g_MainWindow.hDC, g_hRC);

  // Show the window
  ::ShowWindow(hwnd, SW_SHOWDEFAULT);
  ::UpdateWindow(hwnd);

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;   // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;    // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;       // Enable Docking
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;     // Enable Multi-Viewport / Platform Windows

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  //ImGui::StyleColorsClassic();

  // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
  ImGuiStyle& style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

  // Setup Platform/Renderer backends
  ImGui_ImplWin32_InitForOpenGL(hwnd);
  ImGui_ImplOpenGL3_Init();

  // Win32+GL needs specific hooks for viewport, as there are specific things needed to tie Win32 and GL api.
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    IM_ASSERT(platform_io.Renderer_CreateWindow == NULL);
    IM_ASSERT(platform_io.Renderer_DestroyWindow == NULL);
    IM_ASSERT(platform_io.Renderer_SwapBuffers == NULL);
    IM_ASSERT(platform_io.Platform_RenderWindow == NULL);
    platform_io.Renderer_CreateWindow = Hook_Renderer_CreateWindow;
    platform_io.Renderer_DestroyWindow = Hook_Renderer_DestroyWindow;
    platform_io.Renderer_SwapBuffers = Hook_Renderer_SwapBuffers;
    platform_io.Platform_RenderWindow = Hook_Platform_RenderWindow;
  }

  SetWindowIcon(hwnd);

  // Initialize the game and board on screen
  game_pointer = game;
  game->gui_controlled = true;
  viewport.CalculateCoordinates(game_pointer);
  viewport.NewMap(game_pointer);

  app_info.num_players = game->num_players;

//  for (int player_i = 0; player_i < game->num_players; player_i++) {
//    CheckAvailableTypes(game_pointer, player_i);
//    show_player_window[player_i] = true;
//  }
}

void App::Refresh() {
  std::mutex mutex;

  // Poll and handle messages (inputs, window resize, etc.)
  // See the WndProc() function below for our to dispatch events to the Win32 backend.
  while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
  {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
    if (msg.message == WM_QUIT)
      done = true;
  }

  // Start the Dear ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplWin32_NewFrame();
  ImGui::NewFrame();

  mutex.lock();
  int num_players = game_pointer->num_players;
  mutex.unlock();

  if (ImGui::BeginMainMenuBar())
  {
    if (ImGui::BeginMenu("File"))
    {
      ImGui::MenuItem("Show Demo Window", NULL, &show_demo_window);
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("AI"))
    {
      ImGui::MenuItem("AI Menu", NULL, &show_ai_window);
      ImGui::MenuItem("Replay Menu", NULL, &show_replay_window);
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Debug"))
    {
      ImGui::MenuItem("Board", NULL, &show_board_window);
      ImGui::MenuItem("Game", NULL, &show_game_window);

      for (int player_i = 0; player_i < num_players; player_i++) {
        ImGui::MenuItem((std::string("Player ") + std::to_string(player_i + 1)).c_str(), NULL, &show_player_window[player_i]);
      }

      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }

  // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
  if (show_demo_window)
    ImGui::ShowDemoWindow(&show_demo_window);

  // AI Menu
  if (show_ai_window) {
    ImGui::Begin("AI Menu", &show_ai_window);
    window_ai.show(game_pointer, &app_info);

    ImGui::End();
  }

  // Replay Menu
  if (show_replay_window) {
    ImGui::Begin("Replay Menu", &show_replay_window);
    window_replay.show(game_pointer, &viewport, &app_info);

    ImGui::End();
  }

  // Board Menu
  if (show_board_window && app_info.state != AppState::Training) {
    ImGui::Begin("Board", &show_board_window);
    WindowBoard(game_pointer, &viewport);

    ImGui::End();
  }

  // Game Menu
  if (show_game_window && app_info.state != AppState::Training) {
    ImGui::Begin("Game", &show_game_window);
    WindowGame(game_pointer);

    ImGui::End();
  }

  // Player Windows
  for (int player_i = 0; player_i < num_players; player_i++) {
    if (show_player_window[player_i] && app_info.state != AppState::Training) {
      std::string player_string = std::string("Player ") + std::to_string(player_i + 1) + " - " + color_names[player_i];
      ImGui::Begin((player_string).c_str(), &show_player_window[player_i]);
      WindowPlayer(game_pointer, &viewport, player_i, &app_info);

      ImGui::End();
    }
  }

  // Rendering
  ImGui::Render();
  glViewport(0, 0, g_Width, g_Height);
  glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
  glClear(GL_COLOR_BUFFER_BIT);

  if (app_info.state != AppState::Training) {
    viewport.Refresh(game_pointer);
  }

  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  // Update and Render additional Platform Windows
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable && !done)
  {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();

    // Restore the OpenGL rendering context to the main window DC, since platform windows might have changed it.
    wglMakeCurrent(g_MainWindow.hDC, g_hRC);
  }

  // Present
  ::SwapBuffers(g_MainWindow.hDC);
}

App::~App(){
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();

  // Cleanup OpenGL resources
  if (g_hRC) {
    wglMakeCurrent(nullptr, nullptr);  // Unbind the OpenGL rendering context
    wglDeleteContext(g_hRC);           // Delete the OpenGL rendering context
    g_hRC = nullptr;
  }

  // Cleanup Device Context
  if (g_MainWindow.hDC) {
    ::ReleaseDC(hwnd, g_MainWindow.hDC);
    g_MainWindow.hDC = nullptr;
  }

  // Destroy the window
  if (hwnd) {
    ::DestroyWindow(hwnd);
    hwnd = nullptr;
  }

  // Unregister the window class
  if (wc.hInstance) {
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
    wc.hInstance = nullptr;
  }
}

// Helper functions
bool App::CreateDeviceWGL(HWND hWnd, WGL_WindowData* data)
{
  HDC hDc = ::GetDC(hWnd);
  PIXELFORMATDESCRIPTOR pfd = { 0 };
  pfd.nSize = sizeof(pfd);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 32;

  const int pf = ::ChoosePixelFormat(hDc, &pfd);
  if (pf == 0)
    return false;
  if (::SetPixelFormat(hDc, pf, &pfd) == FALSE)
    return false;
  ::ReleaseDC(hWnd, hDc);

  data->hDC = ::GetDC(hWnd);
  if (!g_hRC)
    g_hRC = wglCreateContext(data->hDC);
  return true;
}

void App::CleanupDeviceWGL(HWND hWnd, WGL_WindowData* data)
{
  wglMakeCurrent(nullptr, nullptr);
  ::ReleaseDC(hWnd, data->hDC);
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
LRESULT WINAPI App::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
  if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
    return true;

  switch (msg)
  {
    case WM_SIZE:
      if (wParam != SIZE_MINIMIZED)
      {
        g_Width = LOWORD(lParam);
        g_Height = HIWORD(lParam);
      }
      return 0;
    case WM_SYSCOMMAND:
      if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
        return 0;
      break;
    case WM_DESTROY:
      ::PostQuitMessage(0);
      return 0;
  }
  return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
