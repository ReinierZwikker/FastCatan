#include <cstdio>
#include <stdexcept>
#include <iostream>

// HOW TO (de-)serialise
//  neural_web.to_json((std::string) "ai_test.json", (std::filesystem::path) "ais");
//  neural_web.to_string((std::string) "ai_test.ai", (std::filesystem::path) "ais");
//
//  NeuralWeb test = NeuralWeb((std::string) "ai_test.ai", (std::filesystem::path) "ais");
//
//  std::string test_str = test.to_string();
//
//
//  NeuralWeb test2 = NeuralWeb(test_str);
//  std::string test_str2 = test2.to_string();
//  std::string test_str3 = test2.to_string();
//
//  NeuralWeb test3 = NeuralWeb(test_str2, test_str3, 5342);

#include "ai_zwik_player.h"

AIZwikPlayer::AIZwikPlayer(Player *connected_player) :
        neural_web(amount_of_neurons,
              amount_of_env_inputs+amount_of_inputs,
                   amount_of_outputs,
                   5454321){


  player = connected_player;
  console_tag = color_name(connected_player->player_color) + "> " + color_offset(connected_player->player_color);
//  player_print("Hello World! I am player number " + std::to_string(color_index(player->player_color) + 1) + "!\n");

  update_environment();

  mistakes_made = 0;

}

AIZwikPlayer::AIZwikPlayer(Player *connected_player, const std::string& ai_str) :
        neural_web(ai_str){

  player = connected_player;
  console_tag = color_name(connected_player->player_color) + "> " + color_offset(connected_player->player_color);
//  player_print("Hello World! I am player number " + std::to_string(color_index(player->player_color) + 1) + "!\n");

  update_environment();

  mistakes_made = 0;

}

void AIZwikPlayer::update_environment() {
  Board *board = player->board;

  int env_input_i = 0;
  for (auto tile : board->tile_array) {
    // Make an 0.0 or 1.0 input for each tile type of the tile
    for (int tile_type_i = 0; tile_type_i < 6; ++tile_type_i) {
      inputs[env_input_i++] = (float) (tile.type == index_tile(tile_type_i));
    }
    // Split number tokens into two inputs, scaled from
    //  2 -> 6 : a = 0.3 -> 1.0, b = 0
    // 12 -> 8 : a = 0         , b = 0.3 -> 1.0
    if (tile.number_token < 7) {
      inputs[env_input_i++] = ((float) tile.number_token) / 6;
      inputs[env_input_i++] = 0;
    } else {
      inputs[env_input_i++] = 0;
      inputs[env_input_i++] = (14 - (float) tile.number_token) / 6;
    }
  }

  if (env_input_i != amount_of_env_inputs) {
    throw std::invalid_argument("Not enough environmental inputs!");
  }
}

void AIZwikPlayer::player_print(const std::string& text) {
  printf("%s%s", console_tag.c_str(), text.c_str());
}


int quick_max_index(const float *values, int amount_of_values) {
  // TODO make quick
  int max_index = 0;
  float max_value = 0;
  for (int i = 0; i < amount_of_values; ++i) {
    if (values[i] > max_value) {
      max_index = i;
      max_value = values[i];
    }
  }
  return max_index;
}

void quick_max_three_indices(const float *values, int amount_of_values, int *max_indices) {
  // TODO make quick
  float max_values[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < amount_of_values; ++i) {
    if (values[i] > max_values[0]) {
      max_values[2] = max_values[1];
      max_values[1] = max_values[0];
      max_values[0] = values[i];
      max_indices[2] = max_indices[1];
      max_indices[1] = max_indices[0];
      max_indices[0] = i;
    }
  }
}


Move AIZwikPlayer::get_move(Board *board, int cards[5], GameInfo game_info) {

  /******************
   *     INPUTS     *
   ******************/


  int input_i = amount_of_env_inputs;

  for (auto tile : board->tile_array) {
    inputs[input_i++] = (float) tile.robber;
  }
  for (auto corner : board->corner_array) {
    // Corner : nothing = 0, village = 0.5, city = 1
    inputs[input_i++] = ((float) corner.occupancy) / 2;
    // 4 inputs, 1 if assigned color else 0
    for (int color_i = 0; color_i < 4; ++color_i) {
      inputs[input_i++] = (float) (corner.color == index_color(color_i));
    }
  }
  for (auto street : board->street_array) {
    // 4 inputs, 1 if assigned color else 0
    for (int color_i = 0; color_i < 4; ++color_i) {
      inputs[input_i++] = (float) (street.color == index_color(color_i));
    }
  }

  for (auto card : player->cards) {
    inputs[input_i++] = ((float) card) / 10.0f;
  }

  for (auto development_card : player->dev_cards) {
    inputs[input_i++] = ((float) development_card) / 5.0f;
  }


  for (auto harbor : player->available_harbors) {
    inputs[input_i++] = (float) harbor;
  }

  inputs[input_i++] = (float) player->knight_leader;
  inputs[input_i++] = (float) player->road_leader;

  inputs[input_i++] = ((float) player->victory_points) / 10.0f;

  inputs[input_i++] = ((float) player->resources_left[0]) / 15.0f;
  inputs[input_i++] = ((float) player->resources_left[1]) / 5.0f;
  inputs[input_i++] = ((float) player->resources_left[2]) / 4.0f;

  inputs[input_i++] = ((float) game_info.current_round) / 500.0f;
  for (int turn_type_i = 0; turn_type_i < 9; ++turn_type_i) {
    inputs[input_i++] = (float) (game_info.turn_type == index_turn(turn_type_i));
  }


  if (input_i != amount_of_env_inputs+amount_of_inputs) {
    throw std::invalid_argument("Not enough inputs! "
                                + std::to_string(input_i)
                                + " != "
                                + std::to_string(amount_of_env_inputs+amount_of_inputs) + "\n");
  }

  /******************
   *    RUN WEB     *
   ******************/

  int cycles_ran = neural_web.run_web(&inputs[0], &outputs[0], 10000);

  /******************
   *    OUTPUTS     *
   ******************/

  Move chosen_move;

  // If no valid move found, try 2 times more
  for (int run_try_i = 0; run_try_i < 3; ++run_try_i) {

    int output_i = 0;

    float *move_selections = &outputs[output_i];
    output_i += 10;
    float *street_selections = &outputs[output_i];
    output_i += amount_of_streets;
    float *corner_selections = &outputs[output_i];
    output_i += amount_of_corners;
    float *tile_selections = &outputs[output_i];
    output_i += amount_of_tiles;
    float *tx_card_selections = &outputs[output_i];
    output_i += 5;
    float *rx_card_selections = &outputs[output_i];
    output_i += 5;
    float *dev_card_selections = &outputs[output_i];

    if (output_i != amount_of_outputs) {
      throw std::invalid_argument("Not enough outputs!"
                                  + std::to_string(output_i)
                                  + " != "
                                  + std::to_string(amount_of_outputs) + "\n");
    }

//    Move chosen_move;

/*
 * "ONLY MAX VALUE COUNTS"-METHOD
    chosen_move.type = index_move(quick_max_index(move_selections, 10));

    switch (chosen_move.type) {
      case MoveType::buildStreet:
        chosen_move.index = quick_max_index(street_selections, amount_of_streets);
        break;
      case MoveType::buildVillage:
        chosen_move.index = quick_max_index(corner_selections, amount_of_corners);
        break;
      case MoveType::buildCity:
        chosen_move.index = quick_max_index(corner_selections, amount_of_corners);
        break;
      case MoveType::buyDevelopment:
        chosen_move.index = quick_max_index(dev_card_selections, 5);
        break;
      case MoveType::playDevelopment:
        chosen_move.index = quick_max_index(dev_card_selections, 5);
        break;
      case MoveType::Trade:
        // TODO implement trade
        break;
      case MoveType::Exchange:
        chosen_move.tx_card = index_card(quick_max_index(tx_card_selections, 5));
        chosen_move.tx_amount = 4;
        chosen_move.rx_card = index_card(quick_max_index(rx_card_selections, 5));
        chosen_move.rx_amount = 1;
        break;
      case MoveType::moveRobber:
        chosen_move.index = quick_max_index(tile_selections, amount_of_tiles);
        break;
      case MoveType::getCardBank:
        chosen_move.rx_card = index_card(quick_max_index(rx_card_selections, 5));
        break;
      case MoveType::endTurn:
        break;
      default:
        throw std::invalid_argument("Not a valid move!");
    }

    for (int move_i = 0; move_i < max_available_moves; ++move_i) {
      if (player->available_moves[move_i].type == MoveType::NoMove) {
        break;
      }
      if (chosen_move == player->available_moves[move_i]) {
        //player_print("Found move: " + move2string(chosen_move) + " in " + std::to_string(cycles_ran) + " cycles\n");
        neural_web.clear_queue();
        return chosen_move;
      }
    }
  */

    // "LOOK AT TOP 3"-METHOD
    int actual_available_moves;
    for (actual_available_moves = 0; actual_available_moves < max_available_moves; ++actual_available_moves) {
      if (player->available_moves[actual_available_moves].type == MoveType::NoMove) {
        break;
      }
    }

    int best_move_types[3]{};
    quick_max_three_indices(move_selections, 10, best_move_types);

    bool move_found = false;
    int best_move_i = 0;
    while (!move_found) {
      MoveType best_move_type = index_move(best_move_types[best_move_i]);
      for (int move_i = 0; move_i < actual_available_moves; ++move_i) {
        if (best_move_type == player->available_moves[move_i].type) {
          move_found = true;
          break;
        }
      }
      if (!move_found) {
        ++best_move_i;
      }
      if (best_move_i > 2) {
        break;
      }
    }

    // Continue if move found, else go to fallback
    if (move_found) {
      chosen_move.type = index_move(best_move_types[best_move_i]);

      switch (chosen_move.type) {
        case MoveType::buildStreet:
          chosen_move.index = quick_max_index(street_selections, amount_of_streets);
          break;
        case MoveType::buildVillage:
          chosen_move.index = quick_max_index(corner_selections, amount_of_corners);
          break;
        case MoveType::buildCity:
          chosen_move.index = quick_max_index(corner_selections, amount_of_corners);
          break;
        case MoveType::buyDevelopment:
          chosen_move.index = quick_max_index(dev_card_selections, 5);
          break;
        case MoveType::playDevelopment:
          chosen_move.index = quick_max_index(dev_card_selections, 5);
          break;
        case MoveType::Trade:
          // TODO implement trade
          break;
        case MoveType::Exchange:
          chosen_move.tx_card = index_card(quick_max_index(tx_card_selections, 5));
          chosen_move.tx_amount = 4;
          chosen_move.rx_card = index_card(quick_max_index(rx_card_selections, 5));
          chosen_move.rx_amount = 1;
          break;
        case MoveType::moveRobber:
          chosen_move.index = quick_max_index(tile_selections, amount_of_tiles);
          break;
        case MoveType::getCardBank:
          chosen_move.rx_card = index_card(quick_max_index(rx_card_selections, 5));
          break;
        case MoveType::endTurn:
          break;
        default:
          throw std::invalid_argument("Not a valid move!");
      }

      for (int move_i = 0; move_i < actual_available_moves; ++move_i) {
        if (chosen_move == player->available_moves[move_i]) {
          neural_web.clear_queue();
          // player_print("Found move: " + move2string(chosen_move)
          //              + " with index: " + std::to_string(chosen_move.index)
          //              + " in " + std::to_string(cycles_ran) + " cycles\n");
          return player->available_moves[move_i];
        }
      }

      for (int move_i = 0; move_i < actual_available_moves; ++move_i) {
        if (player->available_moves[move_i].type == chosen_move.type) {
          neural_web.clear_queue();
          //player_print("Found move type: " + move2string(chosen_move) + " in " + std::to_string(cycles_ran) + " cycles\n");
          ++mistakes_made;
          return player->available_moves[move_i];
        }
      }
    }

    // Run some more to find a possible solution
    cycles_ran += neural_web.run_web(&inputs[0], &outputs[0], 3000);
  }

  /******************
   *    FALLBACK    *
   ******************/

  neural_web.clear_queue();


  //for (int move_i = 0; move_i < max_available_moves; ++move_i) {
  //  if (player->available_moves[move_i].type == MoveType::NoMove) {
  //    break;
  //  }
  //}

  mistakes_made += 2;

  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (player->available_moves[move_i].type == MoveType::endTurn) {
      //player_print("No move found!\n      Selected: " + move2string(chosen_move) + "\n      Playing:  " + move2string(player->available_moves[move_i]) + "\n");
      return player->available_moves[move_i];
    }
  }
  //player_print("No move found!\n      Selected: " + move2string(chosen_move) + "\n      Playing:  " + move2string(player->available_moves[0]) + "\n");
  return player->available_moves[0];
}

void AIZwikPlayer::finish_round(Board *board) {

}

AIZwikPlayer::~AIZwikPlayer() = default;
