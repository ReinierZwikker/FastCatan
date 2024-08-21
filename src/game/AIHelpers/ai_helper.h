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

  void delete_players();
  unsigned int number_of_threads;

  std::mutex helper_mutex;

  unsigned int population_size = 0;
  Player*** ai_total_players;
  AISummary** ai_total_summaries = nullptr;
};

#endif //FASTCATAN_AI_HELPER_H
