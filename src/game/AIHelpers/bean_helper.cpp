#include "bean_helper.h"


BeanHelper::BeanHelper(unsigned int pop_size, unsigned int input_seed, unsigned int num_threads) :
                       gen(input_seed), AIHelper(pop_size, num_threads) {

  seed = input_seed;
  survival_amount = (unsigned int)(survival_rate * (float)pop_size);

  BeanNN* bean_nn;
  for (int player_i = 0; player_i < population_size; ++player_i) {
    bean_nn = new BeanNN(false, rd());
    bean_nn->summary.id = player_i;

    nn_vector.push_back(bean_nn);
  }
}

BeanHelper::~BeanHelper() {
  for (int player_i = 0; player_i < population_size; ++player_i) {
    delete nn_vector[player_i];
  }
}

void BeanHelper::update(Game* game, int id) {
  log_game(game, id);

  helper_mutex.lock();
  AIWrapper* ai_players = ai_current_players[id];

  int bean_id;
  for (int player_i = 0; player_i < 4; ++player_i) {
    bean_id = player_i + id * 4;

    ai_players[player_i].player = new Player(&game->board, index_color(player_i));
    auto bean_agent = new BeanPlayer(ai_players[player_i].player, nn_vector[bean_id]->seed);

    bean_agent->neural_net = nn_vector[bean_id];
    ai_players[player_i].player->agent = bean_agent;
    ai_players[player_i].summary = &nn_vector[bean_id]->summary;
  }
  helper_mutex.unlock();
}

void BeanHelper::bubble_sort_indices(int* indices_array, const float* array, int size) {
  for (int i = 0; i < size; ++i) {
    indices_array[i] = i;
  }

  // Bubble sort the indices based on values for MoveType
  for (int i = 0; i < size - 1; ++i) {
    for (int j = 0; j < size - i - 1; ++j) {
      if (array[indices_array[j]] < array[indices_array[j + 1]]) {
        std::swap(indices_array[j], indices_array[j + 1]);
      }
    }
  }
}

void BeanHelper::shuffle_players() {
  helper_mutex.lock();

  std::vector<int> shuffle_indices;
  shuffle_indices.reserve(population_size);

  for (int i = 0; i < population_size; ++i) {
    shuffle_indices.push_back(i);
  }

  std::shuffle(std::begin(shuffle_indices), std::end(shuffle_indices), gen);

  std::vector<BeanNN*> bean_nn_copy = nn_vector;
  for (int i = 0; i < population_size; ++i) {
    nn_vector[i] = bean_nn_copy[shuffle_indices[i]];
  }
  helper_mutex.unlock();
}

void BeanHelper::eliminate() {
  helper_mutex.lock();

  // Calculate the score of the players
  auto* score = new float[population_size];
  for (unsigned int index = 0; index < population_size; ++index) {
    nn_vector[index]->calculate_score();
    score[index] = nn_vector[index]->summary.score;
  }

  // Get the indices of the best part of the population
  int* indices = new int[population_size];
  bubble_sort_indices(indices, score, (int)population_size);

  // Select the players that will survive the elimination
  survived_players = new BeanNN*[survival_amount];
  for (unsigned int i = 0; i < survival_amount; ++i) {
    survived_players[i] = nn_vector[indices[i]];

    if (i < 3) {
      top_players_summaries[i] = survived_players[i]->summary;
      top_player_scores[i] = score[indices[i]];
    }

    survived_players[i]->summary.reset();
  }

  delete[] indices;
  delete[] score;
  helper_mutex.unlock();
}

void BeanHelper::reproduce() {
  std::uniform_int_distribution<int> distribution(0, (int)survival_amount - 1);

  helper_mutex.lock();
  for (int i = 0; i < survival_amount; ++i) {
    nn_vector[i] = survived_players[i];
  }

  for (int i = (int)survival_amount; i < population_size; ++i) {
    BeanNN* parent_1 = survived_players[distribution(gen)];
    BeanNN* parent_2 = survived_players[distribution(gen)];

    auto* child = new BeanNN(parent_1, parent_2, nn_vector[i]);
    nn_vector[i] = child;
  }

  delete[] survived_players;

  helper_mutex.unlock();
}

void BeanHelper::mutate() {
  helper_mutex.lock();
  std::uniform_int_distribution<int> distribution(0, (int)nn_vector[0]->weight_size - mutation_length - 1);

  float min_weight = -1.0f / BeanNN::nodes_per_layer;
  float max_weight = 1.0f / BeanNN::nodes_per_layer;
  std::uniform_real_distribution<float> weight_distribution(min_weight, max_weight);

  for (int nn_i = 0; nn_i < population_size; nn_i += 2) {
    int start_location_mutation = distribution(gen);
    for (int i = 0; i < mutation_length; ++i) {
      nn_vector[nn_i]->weights[start_location_mutation + 1] = weight_distribution(gen);
    }
  }
  helper_mutex.unlock();
}
