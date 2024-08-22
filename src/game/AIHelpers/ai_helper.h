#ifndef FASTCATAN_AI_HELPER_H
#define FASTCATAN_AI_HELPER_H

#include <iostream>

#include "../player.h"
#include "../components.h"
#include "game/game.h"

class AIHelper {
public:
  AIHelper(unsigned int, unsigned int num_threads);
  ~AIHelper();

  void log_game(Game* game, int id, int game_i);

  void delete_players();
  unsigned int number_of_threads;
  unsigned int population_size;

  std::mutex helper_mutex;

  Player*** ai_total_players;
  AISummary** ai_total_summaries;

  AISummary top_players[3] = {AISummary(), AISummary(), AISummary()};
  float top_player_scores[3] = {0.0f, 0.0f, 0.0f};
};

#endif //FASTCATAN_AI_HELPER_H
