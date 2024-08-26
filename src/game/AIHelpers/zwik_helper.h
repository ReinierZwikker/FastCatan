#ifndef FASTCATAN_ZWIK_HELPER_H
#define FASTCATAN_ZWIK_HELPER_H

#include "ai_helper.h"
#include "../AIPlayer/ai_zwik_player.h"

#include <string>

class ZwikHelper : public AIHelper{
public:
  ZwikHelper(unsigned int pop_size, unsigned int num_threads);
  ~ZwikHelper();

  void update(Game*);
  Player* get_new_player(Board* board, Color player_color);


private:
    // TODO: Add AI

  void eliminate();
  void reproduce();
  void mutate();

  std::mt19937 gen;

  std::vector<std::string> ai_nw_genes{};

  struct SelectedAI {
      SelectedAI(Board* board, Color player_color,
                 const std::string& ai_string, int index) {
        player = Player(board, player_color);
        player.agent = new AIZwikPlayer(&player, ai_string);
        str_index = index;
      }
      ~SelectedAI() {
        delete player.agent;
      }

      int str_index;
      Player player;
  };

  std::vector<SelectedAI> ai_current_players{};



};

#endif //FASTCATAN_ZWIK_HELPER_H
