#include "bean_helper.h"

BeanHelper::BeanHelper(unsigned int pop_size, uint8_t num_layers, uint8_t nodes_per_layer,
                       unsigned int seed) : gen(seed), AIHelper(pop_size) {

  for (int player_i = 0; player_i < pop_size; ++player_i) {
    BeanNN bean_player = BeanNN(num_layers, nodes_per_layer, seed);

    bean_players.push_back(&bean_player);
  }
}

BeanHelper::~BeanHelper() {

}

void BeanHelper::update() {

}

void BeanHelper::eliminate() {

}

void BeanHelper::reproduce() {

}

void BeanHelper::mutate() {

}
