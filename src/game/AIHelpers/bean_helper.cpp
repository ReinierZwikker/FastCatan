#include "bean_helper.h"

BeanHelper::BeanHelper(unsigned int pop_size, uint8_t num_layers, uint8_t nodes_per_layer,
                       unsigned int input_seed, unsigned int num_threads) : gen(input_seed),
                       AIHelper(pop_size, num_threads) {

  amount_of_layers = num_layers;
  nodes_in_layer = nodes_per_layer;
  seed = input_seed;

  for (int player_i = 0; player_i < pop_size; ++player_i) {
    auto* bean_nn = new BeanNN(num_layers, nodes_per_layer, seed);
    bean_players.push_back(bean_nn);
  }

}

BeanHelper::~BeanHelper() {
  for (int player_i = 0; player_i < population_size; ++player_i) {
    delete bean_players[player_i];
  }
}

void BeanHelper::update(Game* game, int id) {
  helper_mutex.lock();
  Player** ai_players = ai_total_players[id];
  helper_mutex.unlock();
  for (int player_i = 0; player_i < 4; ++player_i) {
    ai_players[player_i] = new Player(&game->board, index_color(player_i));
    auto bean_agent = new BeanPlayer(ai_players[player_i]);
    bean_agent->neural_net = bean_players[player_i];
    ai_players[player_i]->agent = bean_agent;
  }
}

void BeanHelper::eliminate() {

}

void BeanHelper::reproduce() {

}

void BeanHelper::mutate() {

}
