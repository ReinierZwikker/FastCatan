#include "ai_bean_player.h"
#include <chrono>
#include "cuda_nn.cuh"

/*************
 *    NN     *
 *************/

BeanNN::BeanNN(bool cuda, unsigned int input_seed) : gen(input_seed) {

  seed = input_seed;
  cuda_active = cuda;

  float min_weight = -1.0f / BeanNN::nodes_per_layer;
  float max_weight = 1.0f / BeanNN::nodes_per_layer;
  std::uniform_real_distribution<float> distribution(min_weight, max_weight);

  // Initialize the weights
  weight_size = nodes_per_layer * (input_nodes + nodes_per_layer * (num_hidden_layers - 1) + output_nodes);
  weights = new float[weight_size];
  for (int weight_i = 0; weight_i < weight_size; ++weight_i) {
    // Randomly initialize the weights of the NN
    if (seed != 0) {
      weights[weight_i] = distribution(gen);;
    }
    else {
      // For verification
      weights[weight_i] = (float)weight_i;
    }
  }

  // Initialize biases
  bias_size = nodes_per_layer * num_hidden_layers;
  biases = new float[bias_size];
  for (int bias_i = 0; bias_i < bias_size; ++bias_i) {
    biases[bias_i] = 0.005;  // Init at 0.2, this is just randomly chosen
  }

  if (cuda_active) {
    // ## Initialize CUDA
    // Malloc
    cudaMalloc(&device_weights, weight_size * sizeof(float));
    cudaMalloc(&device_biases, bias_size * sizeof(float));
    cudaMalloc(&device_input, input_nodes * sizeof(float));
    cudaMalloc(&device_output, output_nodes * sizeof(float));

    cudaMalloc(&device_layer_1, nodes_per_layer * sizeof(float));
    cudaMalloc(&device_layer_2, nodes_per_layer * sizeof(float));

    cudaMallocHost(&host_output, output_nodes *sizeof(float));

    // Memcpy
    cudaMemcpy(device_weights, weights, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_biases, biases, bias_size * sizeof(float), cudaMemcpyHostToDevice);
  }
}

BeanNN::BeanNN(const BeanNN* parent_1, const BeanNN* parent_2, const BeanNN* original) {
  seed = original->seed;
  gen = original->gen;

  summary.id = original->summary.id;

  cuda_active = original->cuda_active;

  weight_size = original->weight_size;
  bias_size = original->bias_size;

  // Put the first half of genes from parent 1 and the second half from parent 2
  int half_way_weight = (int)((float)weight_size * 0.5);
  weights = new float[weight_size];
  for (int weight_i = 0; weight_i < weight_size; ++weight_i) {
    if (weight_i < half_way_weight) {
      weights[weight_i] = parent_1->weights[weight_i];
    }
    else {
      weights[weight_i] = parent_2->weights[weight_i];
    }
  }

  int half_way_bias = (int)((float)bias_size * 0.5);
  biases = new float[bias_size];
  for (int bias_i = 0; bias_i < bias_size; ++bias_i) {
    if (bias_i < half_way_bias) {
      biases[bias_i] = parent_1->biases[bias_i];
    }
    else {
      biases[bias_i] = parent_2->biases[bias_i];
    }
  }
}

BeanNN::~BeanNN() {
  if (cuda_active) {
    cudaFree(device_weights);
    cudaFree(device_biases);
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_layer_1);
    cudaFree(device_layer_2);

    cudaFreeHost(host_output);
  }

  delete[] weights;
  delete[] biases;
}

void BeanNN::calculate_score() {
  summary.score = summary.average_points * average_points_mult +
                 (summary.win_rate - 0.25f) * win_rate_mult -
                  summary.average_rounds * average_moves_mult;
}

float* BeanNN::calculate_move_probability(float* input, cudaStream_t* cuda_stream) {
//  auto start = std::chrono::high_resolution_clock::now();

  int weight_offset = 0;
  int bias_offset = 0;

  for (int connection = 0; connection < num_hidden_layers + 1; ++connection) {
    if (connection == 0) {  // Input layer
      cudaMemcpy(device_input, input, input_nodes * sizeof(float), cudaMemcpyHostToDevice);
      step_feed_forward(device_input, device_weights, device_biases,
                        nodes_per_layer, input_nodes, device_layer_1, cuda_stream);
    }
    else if (connection == num_hidden_layers) {  // Output layer
      weight_offset = nodes_per_layer * (input_nodes + nodes_per_layer * (connection - 1));
      bias_offset = nodes_per_layer * connection;

      if (connection % 2 == 0) {
        step_feed_forward(device_layer_2, device_weights + weight_offset, device_biases + bias_offset,
                          output_nodes, nodes_per_layer, device_output, cuda_stream);
      }
      else {
        step_feed_forward(device_layer_1, device_weights + weight_offset, device_biases + bias_offset,
                          output_nodes, nodes_per_layer, device_output, cuda_stream);
      }

      cudaMemcpyAsync(host_output, device_output, output_nodes * sizeof(float), cudaMemcpyDeviceToHost);
      cudaStreamSynchronize(*cuda_stream);

    }
    else {  // Hidden Layer
      weight_offset = nodes_per_layer * (input_nodes + nodes_per_layer * (connection - 1));
      bias_offset = nodes_per_layer * connection;

      if (connection % 2 == 0) {
        step_feed_forward(device_layer_2, device_weights + weight_offset, device_biases + bias_offset,
                          nodes_per_layer, nodes_per_layer, device_layer_1, cuda_stream);
        cudaMemset(device_layer_2, 0, nodes_per_layer * sizeof(float));
      }
      else {
        step_feed_forward(device_layer_1, device_weights + weight_offset, device_biases + bias_offset,
                          nodes_per_layer, nodes_per_layer, device_layer_2, cuda_stream);
        cudaMemset(device_layer_1, 0, nodes_per_layer * sizeof(float));
      }
    }
  }

//  auto end = std::chrono::high_resolution_clock::now();
//  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // Duration in microseconds
//  std::cout << "Function took " << duration.count() << " microseconds" << std::endl;

  return host_output;
}

float* BeanNN::calculate_move_probability(const float* input) {
  static float output[nodes_per_layer];

  int layer_id = 0;
  int old_layer_id = 0;

  float layers[nodes_per_layer * num_hidden_layers];
  int weight_id = 0;
  for (int connection = 0; connection < num_hidden_layers + 1; ++connection) {
    if (connection == 0) {
      // First connection
      for (int node_out = 0; node_out < nodes_per_layer; ++node_out) {
        layers[node_out] = 0;
        for (int node_in = 0; node_in < BeanNN::input_nodes; ++node_in) {

          layers[node_out] += input[node_in] * weights[weight_id++];
        }
        layers[node_out] += biases[get_bias_id(connection, node_out)];
        layers[node_out] = relu(layers[node_out]);
      }
    }
    else if (connection == num_hidden_layers) {
      // Final connection
      for (int node_out = 0; node_out < BeanNN::output_nodes; ++node_out) {
        output[node_out] = 0;
        for (int node_in = 0; node_in < nodes_per_layer; ++node_in) {
          old_layer_id = nodes_per_layer * (connection - 1) + node_in;

          output[node_out] += layers[old_layer_id] * weights[weight_id++];
        }
        output[node_out] = relu(output[node_out]);
      }
    }
    else {
      // Between hidden layers
      for (int node_out = 0; node_out < nodes_per_layer; ++node_out) {
        layer_id = nodes_per_layer * connection + node_out;
        layers[layer_id] = 0;
        for (int node_in = 0; node_in < nodes_per_layer; ++node_in) {
          old_layer_id = nodes_per_layer * (connection - 1) + node_in;

          layers[layer_id] += layers[old_layer_id] * weights[weight_id++];
        }
        layers[layer_id] += biases[layer_id];
        layers[layer_id] = relu(layers[layer_id]);
      }
    }
  }

  return output;
}


int BeanNN::get_weight_id(uint8_t connection, uint8_t node_in, uint8_t node_out) const {
  if (connection == 0) {
    return input_nodes * node_out + node_in;
  }
  else {
    return nodes_per_layer * (input_nodes + nodes_per_layer * (connection - 1) + node_out) + node_in;
  }
}

int BeanNN::get_bias_id(uint8_t connection, uint8_t node_out) const {
  return connection * nodes_per_layer + node_out;
}

float BeanNN::relu(float value) {
  return std::max(0.0f, value);
}

/****************
 *    Agent     *
 ****************/

BeanPlayer::BeanPlayer(Player *connected_player, unsigned int input_seed) {
  agent_seed = input_seed;
  player = connected_player;
  console_tag = color_name(connected_player->player_color) + "> " + color_offset(connected_player->player_color);
}

BeanPlayer::~BeanPlayer() {

}

void BeanPlayer::player_print(std::string text) {
  printf("%s%s", console_tag.c_str(), text.c_str());
}

void BeanPlayer::go_through_moves(MoveType move_type, uint16_t index, CardType tx_card, CardType rx_card) {
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (player->available_moves[move_i].type == MoveType::NoMove) {
      ++chosen_move.mistakes;
      break;  // Break if it hits the end of the available moves
    }
    if (player->available_moves[move_i].type == move_type && player->available_moves[move_i].index == index &&
        player->available_moves[move_i].tx_card == tx_card && player->available_moves[move_i].rx_card == rx_card) {
      player->available_moves[move_i].mistakes = chosen_move.mistakes;
      chosen_move = player->available_moves[move_i];
    }
  }
}

void BeanPlayer::bubble_sort_indices(int* indices_array, const float* array, int size) {
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

/*
 * Gets a valid playing move by:
 * [1] Making a boolean list with all the possible MoveTypes
 * [2] Mapping the Board/Player/Game elements to a single input float array
 * [3] Running the NN and receiving the output float array
 * [4] Sorting the MoveTypes and thus getting a list of which MoveType the AI likes the most
 * [5] Looping through all the possible MoveTypes and then
 *    [a] Getting the maximum index this MoveType allows
 *    [b] IF this MoveType uses indices THEN find the highest ranked index that is available
 *    [c] ELSE IF this is not an Exchange THEN find the first occurrence of this MoveType (buyDevelopment/endTurn)
 *    [d] ELSE find the highest ranked Exchange deal
 * [6] Return the selected move
 */
Move BeanPlayer::get_move(Board *board, int *cards, GameInfo game_info) {
  chosen_move = Move();

  // TODO: Check if available_move_types inits as false!!
  // [1] Gets which move types are available from the possible moves
  bool available_move_types[12] = {false};
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (!available_move_types[(int)player->available_moves[move_i].type]) {
      available_move_types[(int)player->available_moves[move_i].type] = true;
    }
    if (player->available_moves[move_i].type == MoveType::NoMove) {
      break;
    }
  }

  float input[BeanNN::input_nodes];
  float* output;

  // [2] Fill the input array with elements from the Board/Player/Game
  int input_i = 0;
  // Board mapping
  for (Tile tile : board->tile_array) {
    input[input_i++] = (float) tile.type / 6.0f;
    input[input_i++] = (float) tile.number_token / 12.0f;
  }
  for (Corner corner : board->corner_array) {
    input[input_i++] = (float) corner.occupancy / 3.0f;
    input[input_i++] = (float) corner.color / 5.0f;
  }
  for (Street street : board->street_array) {
    input[input_i++] = (float) street.color / 5.0f;
  }
  // Player Mapping
  for (bool available_harbor : player->available_harbors) {
    input[input_i++] = (float) available_harbor;
  }
  for (int card_amount : player->cards) {
    input[input_i++] = (float) card_amount / 20.0f;
  }
  for (int dev_card_amount : player->dev_cards) {
    input[input_i++] = (float) dev_card_amount / 10.0f;
  }
  for (int resource : player->resources_left) {
    input[input_i++] = (float) resource / 15.0f;
  }
  // Game Mapping
  input[input_i++] = (float) game_info.current_dev_card / (float)amount_of_development_cards;
  input[input_i++] = (float) game_info.turn_type / 10.0f;
  input[input_i++] = (float) game_info.current_round / 500.0f;

  if (input_i >= BeanNN::input_nodes) {
    throw std::invalid_argument("Bean Input bigger than allowed");
  }

  // [3] Run the NN
//  auto start = std::chrono::high_resolution_clock::now();
  if (cuda) {
    output = neural_net->calculate_move_probability(input, cuda_stream);
  }
  else {
    output = neural_net->calculate_move_probability(input);
  }
  float* move_types = output;  // MoveType
  float* index = output + 10;  // Index
  float* tx_cards = output + 82;
  float* rx_cards = output + 87;
//  auto end = std::chrono::high_resolution_clock::now();
//  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // Duration in microseconds
//  std::cout << "Function took " << duration.count() << " microseconds" << std::endl;

  // [4] Sort MoveTypes to find the one the AI liked the most
  int move_type_indices[10];
  bubble_sort_indices(move_type_indices, move_types, 10);

  // [5] Get a valid move
  for (int move_type_indice : move_type_indices) {
    if (available_move_types[move_type_indice]) {
      int max_index = 0;
      bool get_tx_rx = false;
      // [a] Get the maximum index depending on the MoveType
      switch (index_move(move_type_indice)) {
        case MoveType::buildStreet:
          max_index = amount_of_streets;
          break;
        case MoveType::buildVillage:
          max_index = amount_of_corners;
          break;
        case MoveType::buildCity:
          max_index = amount_of_corners;
          break;
        case MoveType::buyDevelopment:
          max_index = 0;
          break;
        case MoveType::playDevelopment:
          max_index = 5;
          break;
        case MoveType::Trade:
          max_index = 0;  // TODO: change if trade is implemented
          break;
        case MoveType::Exchange:
          max_index = 0;
          get_tx_rx = true;
          break;
        case MoveType::moveRobber:
          max_index = amount_of_tiles;
          break;
        default:
          max_index = 0;
          break;
      }

      // [b] Get the highest ranked move with an index
      if (max_index > 0) {
        // Sort Indices to find the one the AI liked the most
        int index_indices[100];
        bubble_sort_indices(index_indices, index, max_index);

        // Go through all the available indices
        for (int index_indice: index_indices) {
          go_through_moves(index_move(move_type_indice), index_indice, CardType::NoCard, CardType::NoCard);
          if (chosen_move.type != MoveType::NoMove) {
            return chosen_move;
          }
        }
      }
      // [c] Find the first MoveType in the available moves (buDevelopment, endTurn)
      else if (!get_tx_rx) {
        go_through_moves(index_move(move_type_indice), 0, CardType::NoCard, CardType::NoCard);
        if (chosen_move.type != MoveType::NoMove) {
          return chosen_move;
        }
      }
      // [d] Find the highest ranked Exchange
      else {
        // Initialize tx indices
        int tx_indices[5];
        bubble_sort_indices(tx_indices, tx_cards, 5);

        // Initialize tx indices
        int rx_indices[5];
        bubble_sort_indices(rx_indices, rx_cards, 5);

        // Go through all the possible RX-TX combinations
        for (int tx_indice : tx_indices) {
          for (int rx_indice : rx_indices) {
            go_through_moves(index_move(move_type_indice), 0,
                             index_card(tx_indice), index_card(rx_indice));
            if (chosen_move.type != MoveType::NoMove) {
              return chosen_move;
            }
          }
        }
      }
    }
    else {
      ++chosen_move.mistakes;
    }
  }

  // [6] Return the selected move
  return chosen_move;
}

void BeanPlayer::add_cuda(cudaStream_t* input_cuda) {
  cuda_stream = input_cuda;
  cuda = true;
}

void BeanPlayer::finish_round(Board *board) {

}
