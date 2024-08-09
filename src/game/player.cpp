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
                                     int previous_corner, Color color) {
  previous_streets.insert(street_id);
  std::set<int> return_route = previous_streets;

  for (Corner* corner : board->street_array[street_id].corners) {
    // Check if not moving back through the same corner and that the corner is not occupied by another player
    if ((corner->color == color || corner->color == NoColor) && corner->id != previous_corner) {
      for (Street* street : corner->streets) {
        // Check if the street is defined, is from the current player and if it is not already counted
        if (street != nullptr && street->color == color &&
            previous_streets.find(street->id) == previous_streets.end() &&
            other_route->find(street->id) == other_route->end()) {

          std::set<int> route = traverse_route(street->id, previous_streets, other_route, corner->id, color);

          if (route.size() > return_route.size()) {
            return_route = route;
          };
        }
      }
    }
  }

  return return_route;
}

void Player::check_longest_route(int street_id, Color color) {
  unsigned int route_length = 0;
  std::set<int> route = {street_id};

  for (Corner* corner : board->street_array[street_id].corners) {
    if (corner->color == color || corner->color == NoColor) {
      std::set<int> local_route {street_id};
      for (Street* street : corner->streets) {
        if (street != nullptr && street->color == color && street->id != street_id) {
          local_route = {street_id};

          std::set<int> traveled_route = traverse_route(street->id, local_route, &route, corner->id, color);

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
  }

  if (route_length + 1 > longest_route) {
    longest_route = route_length + 1;
  }
}

int Player::place_street(int street_id) {
  check_longest_route(street_id, player_color);

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

  // Check all adjacent roads for longest route, to see if the village interrupted the longest road.
  for (auto & street : board->corner_array[corner_id].streets) {
    if (street->color != NoColor) {
      int street_id = street->id;
      check_longest_route(street_id, street->color);
    }
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

Move *Player::update_available_moves(TurnType turn_type, Player *players[4], int current_development_card) {
  int current_move_id = 0;
  Move *current_move;

  // Streets
  if (turn_type == openingTurnStreet || turn_type == normalTurn || turn_type == devTurnStreet) {
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
  if (turn_type == normalTurn) {
    if (resources_for_development() && current_development_card < amount_of_development_cards) {
      current_move = add_new_move(current_move_id);
      if (current_move == nullptr) { return available_moves; }

      current_move->move_type = buyDevelopment;

      ++current_move_id;
    }
    for (int i = 0; i < development_cards.size(); ++i) {
      if (!played_development_card && development_cards[i].type != VictoryPoint &&
          !development_cards[i].bought_this_round) {
        current_move = add_new_move(current_move_id);
        if (current_move == nullptr) { return available_moves; }

        current_move->move_type = playDevelopment;
        current_move->index = i;

        ++current_move_id;
      }
    }
  }

/*  // Trading
  if (turn_type == normalTurn) {
    for (int card_i = 0; card_i < 5; ++card_i) {
      if (cards[card_i] >= 1) {
        for (int card_j = 0; card_j < 5; ++card_j) {
          if (card_i != card_j) {
            for (int player_i = 0; player_i < 4; ++player_i) {
              if (players[player_i]->cards[card_j] >= 1) {
                current_move = add_new_move(current_move_id);
                if (current_move == nullptr) { return available_moves; }

                current_move->move_type = Trade;
                current_move->tx_card = index_card(card_i);
                current_move->tx_amount = 1;
                // TODO implement more than 1to1 trading
                current_move->rx_card = index_card(card_j);
                current_move->rx_amount = 1;
                current_move->other_player = index_color(player_i);

                ++current_move_id;
              }
            }
          }
        }
      }
    }
  }*/

  // Exchanging
  if (turn_type == normalTurn) {
    for (int card_i = 0; card_i < 5; ++card_i) {
      // TODO implement harbors?
      if (cards[card_i] >= 4) {
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

  // Year Of Plenty or Monopoly (Development Card)
  if (turn_type == devTurnYearOfPlenty || turn_type == devTurnMonopoly) {
    for (int card_type_i = 0; card_type_i < 5; ++card_type_i) {
      current_move = add_new_move(current_move_id);
      if (current_move == nullptr) { return available_moves; }

      current_move->move_type = getCardBank;
      current_move->index = card_type_i;

      ++current_move_id;
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

void Player::buy_development(DevelopmentType development_type) {
  DevelopmentCard development_card;
  development_card.type = development_type;
  development_card.bought_this_round = true;

  development_cards.push_back(development_card);
  if (development_type == VictoryPoint) {
    ++victory_cards;
  }

  remove_cards(Grain, 1);
  remove_cards(Wool, 1);
  remove_cards(Ore, 1);
}

void Player::play_development(int development_index) {
  if (development_cards[development_index].type == Knight) {
    ++played_knight_cards;
  }

  development_cards.erase(development_cards.begin() + development_index);
  played_development_card = true;
}

void Player::activate_development_cards() {
  played_development_card = false;
  for (DevelopmentCard & dev_card : development_cards) {
    dev_card.bought_this_round = false;
  }
}

int Player::get_total_amount_of_cards() {
  return cards[0] + cards[1] + cards[2] + cards[3] + cards[4];
}

int Player::check_victory_points() {
  victory_points = (5 - resources_left[1]) + 2 * (4 - resources_left[2]);  // Cities and Villages

  if (road_leader) {
    victory_points += 2;  // Longest trade route
  }

  if (knight_leader) {
    victory_points += 2;  // Most knights activated
  }

  victory_points += victory_cards;  // Victory cards

  return victory_points;
}
