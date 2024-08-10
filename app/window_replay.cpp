#include "window_replay.h"

static std::string input_folder = "logs";

std::vector<std::string> split (const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss (s);
  std::string item;

  while (getline (ss, item, delim)) {
    result.push_back (item);
  }

  return result;
}

WindowReplay::WindowReplay() {

}

WindowReplay::~WindowReplay() {

}

void WindowReplay::show() {

  char* input_folder_char = const_cast<char *>(input_folder.c_str());
  ImGui::InputText("Input Folder", input_folder_char, 50);
  input_folder = input_folder_char;

  ImGui::InputInt("Thread", &thread_id);
  if (thread_id < 1) {
    thread_id = 0;
  }
  else if (thread_id > 12) {
    thread_id = 12;
  }

  ImGui::TableNextColumn();
  if (ImGui::Button("Load")) {
    load_games(input_folder);
  }

  if (ImGui::CollapsingHeader("Games")) {
    if (loaded_games.size() > 0) {
      static ImGuiTableFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable;
      if (ImGui::BeginTable("games", 7, flags)) {
        ImGui::TableSetupColumn("Game", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Rounds");
        ImGui::TableSetupColumn("Moves Played");
        ImGui::TableSetupColumn("Run Time [ms]");
        ImGui::TableSetupColumn("Winner");
        ImGui::TableSetupColumn("Load");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        for (int game_i = 0; game_i < loaded_games.size(); ++game_i) {
          ImGui::TableNextColumn(); ImGui::Text("%i", game_i);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].id);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].rounds);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].moves_played);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].run_time);
          ImGui::TableNextColumn(); ImGui::Text("%s", color_names[color_index(loaded_games[game_i].winner)].c_str());

          ImGui::TableNextColumn();
          char button_label[16];
          sprintf(button_label, "Load##G%i", game_i);
          if (ImGui::SmallButton(button_label)) {
            transfer(input_folder, game_i);
          }

          ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        }
        ImGui::EndTable();
      }
    }
  }

  if (ImGui::CollapsingHeader("Moves")) {
    if (loaded_moves.size() > 0) {
      static ImGuiTableFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable;
      if (ImGui::BeginTable("moves", 9, flags)) {
        ImGui::TableSetupColumn("Move", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthFixed, 200.0f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 30.0f);
        ImGui::TableSetupColumn("Index");
        ImGui::TableSetupColumn("Other Player");
        ImGui::TableSetupColumn("TX Card");
        ImGui::TableSetupColumn("TX Amount");
        ImGui::TableSetupColumn("RX Card");
        ImGui::TableSetupColumn("RX Amount");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        for (int move_i = 0; move_i < loaded_moves.size(); ++move_i) {
          if (loaded_moves[move_i].type == MoveType::Replay) {
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(255, 0, 0, 128)); // Red color with 50% opacity
          }

          ImGui::TableNextColumn(); ImGui::Text("%i", move_i);
          ImGui::TableNextColumn(); ImGui::Text("%s", move2string(loaded_moves[move_i]).c_str());
          ImGui::TableNextColumn(); ImGui::Text("%i", move_index(loaded_moves[move_i].type));
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_moves[move_i].index);
          ImGui::TableNextColumn(); ImGui::Text("%s", color_names[color_index(loaded_moves[move_i].other_player)].c_str());
          ImGui::TableNextColumn(); ImGui::Text("%s", card_names[card_index(loaded_moves[move_i].tx_card)].c_str());
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_moves[move_i].tx_amount);
          ImGui::TableNextColumn(); ImGui::Text("%s", card_names[card_index(loaded_moves[move_i].rx_card)].c_str());
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_moves[move_i].rx_amount);

          ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        }
        ImGui::EndTable();
      }
    }
  }

}

void WindowReplay::load_games(const std::string& folder) {
  std::string path_game = folder + "/GameLog_Thread_" + std::to_string(thread_id) + "_game_summaries.dat";
  FILE* file_game = std::fopen(path_game.c_str(), "rb");

  if (!file_game) {
    throw std::invalid_argument("Could not open file");
  }

  // Find the size of the file
  std::fseek(file_game, 0, SEEK_SET);  // Place cursor at the start

  loaded_games.clear();
  GameSummary game_summary;
  while (std::fread(&game_summary, sizeof(GameSummary), 1, file_game) == 1) {
    loaded_games.push_back(game_summary);
  }

  std::fclose(file_game);
}


void WindowReplay::transfer(const std::string& folder, int game_id) {
  std::string path_moves = folder + "/GameLog_Thread_" + std::to_string(thread_id) + "_moves.dat";
  FILE* file_moves = std::fopen(path_moves.c_str(), "rb");

  if (!file_moves) {
    throw std::invalid_argument("Could not open file");
  }

  std::fseek(file_moves, 0, SEEK_SET);  // Place cursor at the start

  loaded_moves.clear();
  for (int game_i = 0; game_i < game_id + 1; ++game_i) {
    for (int move_i = 0; move_i < loaded_games[game_i].moves_played; ++move_i) {
      Move move;
      std::fread(&move, sizeof(Move), 1, file_moves);

      if (game_i == game_id) {
        loaded_moves.push_back(move);
      }
    }
  }

  std::fclose(file_moves);
}

