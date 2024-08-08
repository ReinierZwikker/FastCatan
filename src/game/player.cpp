#include <iostream>

#include "player.h"


Player::Player(Board *global_board, Color assigned_color) {

  board = global_board;
  player_color = assigned_color;

  available_moves = (Move *) malloc(max_available_moves * sizeof(Move));
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    available_moves[move_i] = Move();
  }
}

Player::~Player() {
  free(available_moves);
}

bool corner_occupied(Corner *corner, Color color) {
  // NoColor returns for all colors, otherwise only specific color
  if (color == NoColor) {
    return corner->occupancy == Village || corner->occupancy == City;
  } else {
    return (corner->occupancy == Village || corner->occupancy == City) && corner->color == color;
  }
}

bool street_available(Street *street, Color color, bool opening_turn) {
  // Check if this is the right village
  bool village_available = true;
  if (opening_turn) {
    for (auto & adjacent_corner : street->corners) {
      for (auto & adjacent_street : adjacent_corner->streets) {
        if (adjacent_street != nullptr) {
          if (adjacent_street->color == color) {
            village_available = false;
          }
        }
      }
    }
  }
  bool adjacent = false;
  for (auto & adjacent_corner : street->corners) {
    if (corner_occupied(adjacent_corner, color)) {
      adjacent = true;
    }
    if (!opening_turn) {
      for (auto & adjacent_street : adjacent_corner->streets) {
        if (adjacent_street != nullptr) {
          if (adjacent_street->color == color) {
            adjacent = true;
          }
        }
      }
    }
  }


  return street->color == NoColor
      && adjacent
      && village_available;
}

bool corner_village_available(Corner *corner, Color color, bool opening_turn) {
  if (corner_occupied(corner, NoColor)) {
    return false;
  }

  for (auto &adjacent_street : corner->streets) {
    if (adjacent_street != nullptr) {
      for (auto adjacent_corner: adjacent_street->corners) {
        if (corner_occupied(adjacent_corner, NoColor)) {
          return false;
        }
      }
    }
  }

  if (opening_turn) {
      return true;
  } else {
    for (auto &adjacent_street: corner->streets) {
      if (adjacent_street != nullptr) {
        if (adjacent_street->color == color) {
          return true;
        }
      }
    }
    return false;
  }
}

bool corner_city_available(Corner *corner, Color color) {
  return corner->occupancy == Village
      && corner->color == color;
}


bool Player::resources_for_street() {
  return cards[card_index(Brick)] >= 1
         && cards[card_index(Lumber)] >= 1
         && resources_left[0] >= 1;
}

bool Player::resources_for_village() {
  return cards[card_index(Brick)] >= 1
         && cards[card_index(Lumber)] >= 1
         && cards[card_index(Grain)] >= 1
         && cards[card_index(Wool)] >= 1
         && resources_left[1] >= 1;
}

bool Player::resources_for_city() {
  return cards[card_index(Ore)] >= 3
         && cards[card_index(Grain)] >= 2
         && resources_left[2] >= 1;
}

bool Player::resources_for_development() {
  return cards[card_index(Ore)] >= 1
         && cards[card_index(Grain)] >= 1
         && cards[card_index(Wool)] >= 1;
}

std::set<int> Player::traverse_route(int street_id, std::set<int> previous_streets, std::set<int>* other_route,
                                     int previous_corner) {
  previous_streets.insert(street_id);
  std::set<int> return_route = previous_streets;

  for (Corner* corner : board->street_array[street_id].corners) {
    for (Street* street : corner->streets) {
      // Check if the street is defined, is from the current player and if it is not already counted
      // Also check if we are not moving back through the same corner
      if (street != nullptr && street->color == player_color && corner->id != previous_corner &&
          previous_streets.find(street->id) == previous_streets.end() &&
          other_route->find(street->id) == other_route->end()) {

        std::set<int> route = traverse_route(street->id, previous_streets, other_route, corner->id);

        if (route.size() > return_route.size()) {
          return_route = route;
        };
      }
    }
  }

  return return_route;
}

unsigned int Player::check_longest_route(int street_id) {
  unsigned int route_length = 0;
  std::set<int> route = {street_id};

  for (Corner* corner : board->street_array[street_id].corners) {
    std::set<int> local_route {street_id};
    for (Street* street : corner->streets) {
      if (street != nullptr && street->color == player_color) {
        local_route = {street_id};

        std::set<int> traveled_route = traverse_route(street->id, local_route, &route, corner->id);

        if (traveled_route.size() > local_route.size()) {
          local_route = traveled_route;
        };
      }
    }
    if (route.size() == 1) {
      route = local_route;  // Replace the main route with the longest local route
    }
    route_length += local_route.size() - 1;
  }

  return route_length + 1;
}

int Player::place_street(int street_id) {
  unsigned int local_longest_route = check_longest_route(street_id);
  if (local_longest_route > longest_route) {
    longest_route = local_longest_route;
  }

  if (street_available(&board->street_array[street_id], player_color, false)
   && resources_for_street()) {
    board->street_array[street_id].color = player_color;
    resources_left[0]--; // Remove one street from pool
    remove_cards(Brick, 1);
    remove_cards(Lumber, 1);
    return 0;
  } else {
    return 1;
  }
}

int Player::place_village(int corner_id) {
  if (corner_village_available(&board->corner_array[corner_id], player_color, true)
   && resources_for_village()) {
    board->corner_array[corner_id].occupancy = Village;
    board->corner_array[corner_id].color = player_color;
    resources_left[1]--; // Remove one village from pool
    remove_cards(Brick, 1);
    remove_cards(Lumber, 1);
    remove_cards(Grain, 1);
    remove_cards(Wool, 1);
    return 0;
  } else {
    return 1;
  }
}

int Player::place_city(int corner_id) {
  if (corner_city_available(&board->corner_array[corner_id], player_color)
   && resources_for_city()) {
    board->corner_array[corner_id].occupancy = City;
    board->corner_array[corner_id].color = player_color;
    resources_left[2]--; // Remove one city from pool
    resources_left[1]++; // Return one village to pool
    remove_cards(Ore, 3);
    remove_cards(Grain, 2);
    return 0;
  } else {
    return 1;
  }
}

Move *Player::add_new_move(int move_id) {
  if (move_id >= max_available_moves) {
    printf("Ran out of available moves!");
    return nullptr;
  }
  available_moves[move_id] = Move();
  return &available_moves[move_id];
}

Move *Player::update_available_moves(TurnType turn_type, Player *players[4]) {
  int current_move_id = 0;
  Move *current_move;

  // Streets
  if (turn_type == openingTurnStreet || turn_type == normalTurn) {
    for (int street_i = 0; street_i < amount_of_streets; ++street_i) {
      if (street_available(&board->street_array[street_i], player_color, turn_type == openingTurnStreet)
       && resources_for_street()) {
        current_move = add_new_move(current_move_id);
        if (current_move == nullptr) { return available_moves; }

        current_move->move_type = buildStreet;
        current_move->index = street_i;

        ++current_move_id;
      }
    }
  }

  // Villages
  if (turn_type == openingTurnVillage || turn_type == normalTurn) {
    for (int corner_i = 0; corner_i < amount_of_corners; ++corner_i) {
      if (corner_village_available(&board->corner_array[corner_i], player_color, turn_type == openingTurnVillage)
       && resources_for_village()) {
        current_move = add_new_move(current_move_id);
        if (current_move == nullptr) { return available_moves; }

        current_move->move_type = buildVillage;
        current_move->index = corner_i;

        ++current_move_id;
      }
    }
  }

  // Cities
  if (turn_type == normalTurn) {
    for (int corner_i = 0; corner_i < amount_of_corners; ++corner_i) {
      if (corner_city_available(&board->corner_array[corner_i], player_color)
       && resources_for_city()) {
        current_move = add_new_move(current_move_id);
        if (current_move == nullptr) { return available_moves; }

        current_move->move_type = buildCity;
        current_move->index = corner_i;

        ++current_move_id;
      }
    }
  }



  // Development Card
  // TODO implement development cards

  // Trading


  // Exchanging
  if (turn_type == normalTurn) {
    for (int card_i = 0; card_i < 5; ++card_i) {
      // TODO implement harbors?
      if (cards[card_i] > 4) {
        for (int card_j = 0; card_j < 5; ++card_j) {
          if (card_i != card_j) {
            current_move = add_new_move(current_move_id);
            if (current_move == nullptr) { return available_moves; }

            current_move->move_type = Exchange;
            current_move->tx_card = index_card(card_i);
            current_move->tx_amount = 4;
            current_move->rx_card = index_card(card_j);
            current_move->rx_amount = 1;

            ++current_move_id;
          }
        }
      }
    }
  }


  // Robber
  if (turn_type == robberTurn) {
    for (int tile_i = 0; tile_i < amount_of_tiles; ++tile_i) {
      if (!board->tile_array[tile_i].robber) {
        current_move = add_new_move(current_move_id);
        if (current_move == nullptr) { return available_moves; }

        current_move->move_type = moveRobber;
        current_move->index = tile_i;

        ++current_move_id;
      }
    }
  }

  // End Turn
  if (turn_type == normalTurn) {

    current_move = add_new_move(current_move_id);
    if (current_move == nullptr) { return available_moves; }

    current_move->move_type = endTurn;

    ++current_move_id;
  }

  // Closing mark
  if (current_move_id >= max_available_moves) {
    printf("Ran out of available moves!");
    return available_moves;
  }
  available_moves[current_move_id] = Move();
  available_moves[current_move_id].move_type = NoMove;
  return available_moves;
}

void Player::set_cards(int brick, int lumber, int ore, int grain, int wool) {
  cards[0] = brick;
  cards[1] = lumber;
  cards[2] = ore;
  cards[3] = grain;
  cards[4] = wool;
}

void Player::add_cards(CardType card_type, int amount) {
  if (card_type != NoCard) {
    cards[card_index(card_type)] += amount;
  }
}

void Player::remove_cards(CardType card_type, int amount) {
  if (card_type != NoCard) {
    cards[card_index(card_type)] -= amount;
  }
}

int Player::get_total_amount_of_cards() {
  return cards[0] + cards[1] + cards[2] + cards[3] + cards[4];
}

int Player::check_victory_points() {
  // TODO implement other sources of VPs
  victory_points = (5 - resources_left[1]) + 2 * (4 - resources_left[2]);
  return victory_points;
}