#ifndef FASTCATAN_ZWIK_HELPER_H
#define FASTCATAN_ZWIK_HELPER_H

#include "ai_helper.h"
#include "../AIPlayer/ai_zwik_player.h"
#include "../player.h"

#include <vector>
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

  struct Individual {
      explicit Individual(int seed) {
        gene = NeuralWeb(AIZwikPlayer::amount_of_neurons,
                         AIZwikPlayer::amount_of_env_inputs + AIZwikPlayer::amount_of_inputs,
                         AIZwikPlayer::amount_of_outputs,
                         seed).to_string();
        score = 0;
      }
      Individual(const std::string& parent_A_gene,
                 const std::string& parent_B_gene) {
        gene = NeuralWeb(parent_A_gene, parent_B_gene).to_string();
        score = 0;
      }

      std::string gene;
      int score;
  };

  std::vector<Individual> gene_pool{};

  struct SelectedAI {
      SelectedAI(Board *board, Color player_color,
                 const std::string& ai_string, int index) {
        player = Player(board, player_color, index);
        player.agent = new AIZwikPlayer(&player, ai_string);
        str_index = index;
      }
      ~SelectedAI() {
        delete player.agent;
      }

      int str_index;
      Player player;
  };

  std::vector<SelectedAI> current_players{};



};

#endif //FASTCATAN_ZWIK_HELPER_H
