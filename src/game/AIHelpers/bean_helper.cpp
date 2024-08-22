#include "bean_helper.h"


BeanHelper::BeanHelper(unsigned int pop_size, unsigned int input_seed, unsigned int num_threads) :
                       gen(input_seed), AIHelper(pop_size, num_threads) {

  seed = input_seed;

  BeanNN* bean_nn;
  for (int player_i = 0; player_i < num_threads * 4; ++player_i) {
    bean_nn = new BeanNN(false, rd());

    bean_nn_vector.push_back(bean_nn);
  }

}

BeanHelper::~BeanHelper() {
  for (int player_i = 0; player_i < number_of_threads; ++player_i) {
    delete bean_nn_vector[player_i];
  }
}

void BeanHelper::update(Game* game, int id, int game_i) {
  log_game(game, id, game_i);

  helper_mutex.lock();
  Player** ai_players = ai_total_players[id];
  helper_mutex.unlock();
  for (int player_i = 0; player_i < 4; ++player_i) {
    ai_players[player_i] = new Player(&game->board, index_color(player_i));
    auto bean_agent = new BeanPlayer(ai_players[player_i], bean_nn_vector[player_i]->seed);
    bean_agent->neural_net = bean_nn_vector[player_i];
    ai_players[player_i]->agent = bean_agent;
  }
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
  const int size = 4;

  std::vector<int> shuffle_indices;
  shuffle_indices.reserve(size);

  for (int i = 0; i < size; ++i) {
    shuffle_indices.push_back(i);
  }

  std::shuffle(std::begin(shuffle_indices), std::end(shuffle_indices), gen);

  int player_i, thread_i, new_player_i, new_thread_i;
  std::vector<BeanNN*> bean_nn_copy = bean_nn_vector;
  AISummary** ai_summary_copy = ai_total_summaries;
  for (int i = 0; i < size; ++i) {
    player_i = i % 4;
    thread_i = i / 4;

    new_player_i = shuffle_indices[i] % 4;
    new_thread_i = shuffle_indices[i] / 4;

    bean_nn_vector[i] = bean_nn_copy[shuffle_indices[i]];
    ai_total_summaries[thread_i][player_i] = ai_summary_copy[new_thread_i][new_player_i];
  }
  helper_mutex.unlock();
}

void BeanHelper::eliminate() {
  std::cout << "ELIMINATE" << std::endl;

  helper_mutex.lock();
  auto* score = new float[number_of_threads * 4];
  for (int thread_i = 0; thread_i < number_of_threads; ++thread_i) {
    for (int player_i = 0; player_i < 4; ++player_i) {
      score[thread_i * 4 + player_i] = ai_total_summaries[thread_i][player_i].average_points * average_points_mult +
                                      (ai_total_summaries[thread_i][player_i].win_rate - 0.25f) * win_rate_mult -
                                       ai_total_summaries[thread_i][player_i].average_rounds * average_moves_mult;
    }
  }

  int* indices = new int[number_of_threads * 4];
  bubble_sort_indices(indices, score, (int)number_of_threads * 4);

  survived_players = new BeanNN*[number_of_threads];
  for (unsigned int i = 0; i < number_of_threads; ++i) {
    survived_players[i] = bean_nn_vector[indices[i]];
  }

  int player, thread;
  for (int i = 0; i < 3; ++i) {
    player = indices[i] % 4;
    thread = indices[i] / 4;
    top_players[i] = ai_total_summaries[thread][player];
    top_player_scores[i] = score[indices[i]];
  }

  for (int thread_i = 0; thread_i < number_of_threads; ++thread_i) {
    for (int player_i = 0; player_i < 4; ++player_i) {
      ai_total_summaries[thread_i][player_i] = AISummary();
    }
  }

  delete[] indices;
  delete[] score;
  helper_mutex.unlock();
}

void BeanHelper::reproduce() {
  std::uniform_int_distribution<int> distribution(0, (int)number_of_threads - 1);

  helper_mutex.lock();
  for (int i = 0; i < number_of_threads; ++i) {
    bean_nn_vector[i] = survived_players[i];
  }

  for (int i = (int)number_of_threads; i < number_of_threads * 4; ++i) {
    BeanNN* parent_1 = survived_players[distribution(gen)];
    BeanNN* parent_2 = survived_players[distribution(gen)];

    int half_way_i = BeanNN::nodes_per_layer * (BeanNN::input_nodes + BeanNN::nodes_per_layer * BeanNN::num_hidden_layers * 0.5);
    bean_nn_vector[i] = parent_1;

    // Add "DNA" from second parent
    for (int j = half_way_i; j < parent_2->weight_size; ++j) {
      bean_nn_vector[i]->weights[j] = parent_2->weights[j];
    }
  }

  delete[] survived_players;

  helper_mutex.unlock();
}

void BeanHelper::mutate() {
  std::uniform_int_distribution<int> distribution(0, (int)bean_nn_vector[0]->weight_size - mutation_length - 1);

  float min_weight = -1.0f / BeanNN::nodes_per_layer;
  float max_weight = 1.0f / BeanNN::nodes_per_layer;
  std::uniform_real_distribution<float> weight_distribution(min_weight, max_weight);

  for (int nn_i = 0; nn_i < number_of_threads * 4; nn_i += 2) {
    int start_location_mutation = distribution(gen);
    for (int i = 0; i < mutation_length; ++i) {
      bean_nn_vector[nn_i]->weights[start_location_mutation + 1] = weight_distribution(gen);
    }

  }

}
