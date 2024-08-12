#include "ai_bean_player.h"

/*************
 *    NN     *
 *************/

BeanNN::BeanNN(uint8_t num_layers, uint16_t nodes_per_row, unsigned int input_seed) : gen(input_seed) {
  seed = input_seed;
  num_hidden_layers = num_layers;
  nodes_per_layer = nodes_per_row;

  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  // Initialize the weights
  weight_size = nodes_per_row * (input_nodes + nodes_per_row * (num_layers - 1) + output_nodes);
  weights = new float[weight_size];
  for (int weight_i = 0; weight_i < weight_size; ++weight_i) {
    // Randomly initialize the weights of the NN
    float random_number = distribution(gen);
    weights[weight_i] = random_number;
  }

  // Initialize biases
  bias_size = input_nodes + nodes_per_layer * num_hidden_layers + output_nodes;
  biases = new float[bias_size];
  for (int bias_i = 0; bias_i < bias_size; ++bias_i) {
    biases[bias_i] = 0.2;  // Init at 0.2, this is just randomly chosen
  }

}

BeanNN::~BeanNN() {
  delete[] weights;
}

float* BeanNN::calculate_move_probability(float* input) {
  static float output[output_nodes];


  return output;
}

int BeanNN::get_weight_id(uint8_t connection, uint8_t node_in, uint8_t node_out) const {
  if (connection == 0) {
    return input_nodes * node_in + node_out;
  }
  else {
    return nodes_per_layer * (input_nodes + nodes_per_layer * (connection - 1) + node_in) + node_out;
  }
}

float BeanNN::relu(float value) {
  return std::max(0.0f, value);
}

/****************
 *    Agent     *
 ****************/

BeanPlayer::BeanPlayer(Player *connected_player) {
  player = connected_player;
  console_tag = color_name(connected_player->player_color) + "> " + color_offset(connected_player->player_color);
}

BeanPlayer::~BeanPlayer() {

}

void BeanPlayer::player_print(std::string text) {
  printf("%s%s", console_tag.c_str(), text.c_str());
}

Move BeanPlayer::go_through_moves(MoveType move_type, uint16_t index, CardType tx_card, CardType rx_card) {
  Move move;
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (player->available_moves[move_i].type == MoveType::NoMove) {
      break;  // Break if it hits the end of the available moves
    }
    if (player->available_moves[move_i].type == move_type && player->available_moves[move_i].index == index &&
        player->available_moves[move_i].tx_card == tx_card && player->available_moves[move_i].rx_card == rx_card) {
      move = player->available_moves[move_i];
    }
  }
  return move;
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
  Move move;

  // TODO: Check if available_move_types inits as false!!
  // [1] Gets which move types are available from the possible moves
  bool available_move_types[12];
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
    input[input_i++] = (float) tile.type;
    input[input_i++] = (float) tile.number_token;
  }
  for (Corner corner : board->corner_array) {
    input[input_i++] = (float) corner.occupancy;
    input[input_i++] = (float) corner.color;
  }
  for (Street street : board->street_array) {
    input[input_i++] = (float) street.color;
  }
  // Player Mapping
  for (bool available_harbor : player->available_harbors) {
    input[input_i++] = (float) available_harbor;
  }
  for (int card_amount : player->cards) {
    input[input_i++] = (float) card_amount;
  }
  for (int dev_card_amount : player->dev_cards) {
    input[input_i++] = (float) dev_card_amount;
  }
  for (int resource : player->resources_left) {
    input[input_i++] = (float) resource;
  }
  // Game Mapping
  input[input_i++] = (float) game_info.current_dev_card;
  input[input_i++] = (float) game_info.turn_type;
  input[input_i++] = (float) game_info.current_round;

  if (input_i >= BeanNN::input_nodes) {
    throw std::invalid_argument("Bean Input bigger than allowed");
  }

  // [3] Run the NN
  output = neural_net->calculate_move_probability(input);
  float* move_types = output;  // MoveType
  float* index = output + 9;  // Index
  float* tx_cards = output + 81;
  float* rx_cards = output + 86;

  // [4] Sort MoveTypes to find the one the AI liked the most
  int move_type_indices[9];
  bubble_sort_indices(move_type_indices, move_types, 9);

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
        int index_indices[max_index];
        bubble_sort_indices(index_indices, index, max_index);

        // Go through all the available indices
        for (int index_indice: index_indices) {
          move = go_through_moves(index_move(move_type_indice), index_indice, index_card(0), index_card(0));
        }
      }
      // [c] Find the first MoveType in the available moves (buDevelopment, endTurn)
      else if (!get_tx_rx) {
        move = go_through_moves(index_move(move_type_indice), 0, index_card(0), index_card(0));
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
            move = go_through_moves(index_move(move_type_indice), 0,
                                    index_card(tx_indice), index_card(rx_indice));
          }
        }
      }
    }
  }

  // [6] Return the selected move
  return move;
}

void BeanPlayer::finish_round(Board *board) {

}
