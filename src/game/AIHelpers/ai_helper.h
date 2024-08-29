#ifndef FASTCATAN_AI_HELPER_H
#define FASTCATAN_AI_HELPER_H

#include <iostream>

#include "../player.h"
#include "../components.h"
#include "game/game.h"

struct AIWrapper {
  PlayerType type = PlayerType::NoPlayer;
  Player* player = nullptr;
  PlayerSummary* summary = nullptr;
};


class AIHelper {
public:
  AIHelper(unsigned int, unsigned int num_threads);
  ~AIHelper();

  void log_game(Game* game, int id);

  void delete_players();
  unsigned int number_of_threads;
  unsigned int population_size;

  std::mutex helper_mutex;

  AIWrapper** ai_current_players;

  PlayerSummary top_players_summaries[3] = {PlayerSummary(), PlayerSummary(), PlayerSummary()};
  float top_player_scores[3] = {0.0f, 0.0f, 0.0f};
};

#endif //FASTCATAN_AI_HELPER_H
