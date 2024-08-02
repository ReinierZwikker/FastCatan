#include "window_player.h"

void WindowPlayer(Game* game, int player_id) {
  if (ImGui::CollapsingHeader("Cards")) {
    if (ImGui::BeginTable("split", 2)) {
      ImGui::TableNextColumn(); ImGui::Text("Resource");
      ImGui::TableNextColumn(); ImGui::Text("Amount");
      ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
      for (int card_type_i = 0; card_type_i < 5; card_type_i++) {
        ImGui::TableNextColumn(); ImGui::Text(card_names_char[card_type_i]);
        ImGui::TableNextColumn(); ImGui::Text("%i", game->players[player_id]->cards[card_type_i]);
      }

      ImGui::EndTable();
    }
  }

}
