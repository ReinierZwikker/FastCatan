#include "ai_helper.h"


AIHelper::AIHelper(unsigned int pop_size, unsigned int num_threads) {
  if (pop_size > pow(2, sizeof(PlayerSummary::id) * 8)) {
    throw std::invalid_argument("Population size is too big");
  }
  population_size = pop_size;
  number_of_threads = num_threads;

  ai_current_players = new AIWrapper*[num_threads];
  for (int thread = 0; thread < num_threads; ++thread) {
    ai_current_players[thread] = new AIWrapper[4];
  }

}

AIHelper::~AIHelper() {
  delete_players();
  for (int thread = 0; thread < number_of_threads; ++thread) {
    delete ai_current_players[thread];
  }
  delete ai_current_players;
}

void AIHelper::log_game(Game* game, int id) {
  helper_mutex.lock();
  for (int player_i = 0; player_i < game->num_players; ++player_i) {
    PlayerSummary* ai_summary = ai_current_players[id][player_i].summary;

    if (ai_summary != nullptr) {
      ++ai_summary->games_played;

      if (ai_summary->games_played > 0) {

        ai_summary->average_rounds = ((ai_summary->average_rounds * (float)(ai_summary->games_played - 1) +
                                                (float)game->current_round) /
                                               ((float)ai_summary->games_played + 1));

        ai_summary->average_points = ((ai_summary->average_points * (float)(ai_summary->games_played - 1) +
                                                (float)game->players[player_i]->victory_points) /
                                               ((float)ai_summary->games_played + 1));

        if (ai_summary->average_points > 11) {
          std::cout << game->players[player_i]->victory_points << std::endl;
          std::cout << ai_summary->games_played << std::endl;
        }

        if (game->game_winner == game->players[player_i]->player_color) {
          ++ai_summary->wins;

        }
        ai_summary->win_rate = (float)ai_summary->wins / ((float)ai_summary->games_played + 1);

        ai_summary->mistakes = ((ai_summary->mistakes * (float)(ai_summary->games_played - 1) +
                                   (float)ai_current_players[id][player_i].player->mistakes) /
                                   ((float)ai_summary->games_played + 1));
      }
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

