#include "ai_helper.h"


AIHelper::AIHelper(unsigned int pop_size, unsigned int num_threads) {
  if (pop_size > pow(2, sizeof(AISummary::id) * 8)) {
    throw std::invalid_argument("Population size is too big");
  }
  population_size = pop_size;
  number_of_threads = num_threads;

  ai_total_players = new Player**[num_threads];
  ai_total_summaries = new AISummary*[num_threads];

  for (int thread = 0; thread < num_threads; ++thread) {
    ai_total_players[thread] = new Player*[4];
    ai_total_summaries[thread] = new AISummary[4];

    for (int player = 0; player < 4; ++player) {
      ai_total_summaries[thread][player].id = thread * 4 + player;
    }
  }
}

AIHelper::~AIHelper() {
  delete_players();
  for (int thread = 0; thread < number_of_threads; ++thread) {
    delete ai_total_players[thread];
    delete ai_total_summaries[thread];
  }
  delete ai_total_players;
  delete ai_total_summaries;
}

void AIHelper::log_game(Game* game, int id, int game_i) {
  helper_mutex.lock();
  AISummary* ai_summaries = ai_total_summaries[id];
  for (int player_i = 0; player_i < game->num_players; ++player_i) {
    if (game_i > 0) {

      ai_summaries[player_i].average_rounds = ((ai_summaries[player_i].average_rounds * (float)(game_i - 1) +
                                               (float)game->current_round) /
                                              ((float)game_i + 1));

      ai_summaries[player_i].average_points = ((ai_summaries[player_i].average_points * (float)(game_i - 1) +
                                               (float)game->players[player_i]->victory_points) /
                                              ((float)game_i + 1));

      if (ai_summaries[player_i].average_points > 11) {
        std::cout << game->players[player_i]->victory_points << std::endl;
        std::cout << game_i << std::endl;
      }

      if (game->game_winner == game->players[player_i]->player_color) {
        ++ai_summaries[player_i].wins;

      }
      ai_summaries[player_i].win_rate = (float)ai_summaries[player_i].wins / ((float)game_i + 1);
    }
  }
  helper_mutex.unlock();
}

void AIHelper::delete_players() {
//  for (int thread = 0; thread < number_of_threads; ++thread) {
//    for (int player_i = 0; player_i < 4; ++player_i) {
//      if (ai_total_players[thread][player_i] != nullptr) {
//        delete ai_total_players[thread][player_i]->agent;
//        delete ai_total_players[thread][player_i];
//      }
//    }
//  }
}

