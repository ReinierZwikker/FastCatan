#include "ai_helper.h"


AIHelper::AIHelper(unsigned int pop_size, unsigned int num_threads) {
  if (pop_size > pow(2, sizeof(AISummary::id) * 8)) {
    throw std::invalid_argument("Population size is too big");
  }
  population_size = pop_size;
  number_of_threads = num_threads;

  ai_total_players = new Player**[num_threads];

  for (int thread = 0; thread < num_threads; ++thread) {
    ai_total_players[thread] = new Player*[4];
  }
}

AIHelper::~AIHelper() {
  delete_players();
  for (int thread = 0; thread < number_of_threads; ++thread) {
    delete ai_total_players[thread];
  }
  delete ai_total_players;
}

void AIHelper::delete_players() {
  for (int thread = 0; thread < number_of_threads; ++thread) {
    for (int player_i = 0; player_i < 4; ++player_i) {
      if (ai_total_players[thread][player_i] != nullptr) {
        delete ai_total_players[thread][player_i]->agent;
        delete ai_total_players[thread][player_i];
      }
    }
  }
}

