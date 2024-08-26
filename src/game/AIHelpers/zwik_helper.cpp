#include "zwik_helper.h"

#include <ctime>

ZwikHelper::ZwikHelper(unsigned int pop_size, unsigned int num_threads) : AIHelper(pop_size, num_threads), gen((int) time(nullptr)) {
  if (pop_size > pow(2, sizeof(AISummary::id) * 8)) {
    throw std::invalid_argument("Population size is too big");
  }
  population_size = pop_size;
  number_of_threads = num_threads;

  gene_pool.reserve(population_size);
  std::uniform_int_distribution<> random_seed(0, 100 * (int) population_size);

  // Start with random gene pool
  for (int pop_i = 0; pop_i < population_size; ++pop_i) {
    gene_pool.emplace_back(random_seed(gen));
  }

  current_players.reserve(number_of_threads);
}

ZwikHelper::~ZwikHelper() {
  delete_players();
  for (int thread = 0; thread < number_of_threads; ++thread) {
    delete ai_total_players[thread];
  }
  delete ai_total_players;
}

void ZwikHelper::update(Game* game) {



  // TODO find player ranking

  // TODO remove genes from worst players

  // TODO add new genes crossed from best parents
  // ..% parents remain, ..% children new

  current_players.clear();


}

Player *ZwikHelper::get_new_player(Board* board, Color player_color) {
  std::uniform_int_distribution<> random_player(0, (int) population_size - 1);
  int selected_ai = random_player(gen);

  current_players.emplace_back(board, player_color, gene_pool[selected_ai], selected_ai);

  return &current_players.back().player;
}

void ZwikHelper::eliminate() {

}

void ZwikHelper::reproduce() {

}

void ZwikHelper::mutate() {

}
