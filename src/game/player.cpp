#include <iostream>

#include "player.h"


Player::Player(Board *global_board, Color assigned_color) {

  board = global_board;
  player_color = assigned_color;

  available_moves = (Move *) malloc(max_available_moves * sizeof(Move));
}

Player::~Player() {
  free(available_moves);
}

int Player::place_street(int street_id) {
  if (board->street_array[street_id].color == NoColor &&
      resources_left[0] > 0) {
    board->street_array[street_id].color = player_color;
    resources_left[0]--; // Remove one street from pool
    return 0;
  } else {
    return 1;
  }
}

int Player::place_village(int corner_id) {
  if (board->corner_array[corner_id].color == NoColor &&
      board->corner_array[corner_id].occupancy == EmptyCorner &&
      resources_left[1] > 0) {
    board->corner_array[corner_id].occupancy = Village;
    board->corner_array[corner_id].color = player_color;
    resources_left[1]--; // Remove one village from pool
    return 0;
  } else {
    return 1;
  }
}

int Player::place_city(int corner_id) {
  if (board->corner_array[corner_id].color == player_color &&
      board->corner_array[corner_id].occupancy == Village &&
      resources_left[2] > 0) {
    board->corner_array[corner_id].occupancy = City;
    board->corner_array[corner_id].color = player_color;
    resources_left[2]--; // Remove one city from pool
    resources_left[1]++; // Return one village to pool
    return 0;
  } else {
    return 1;
  }
}

bool corner_occupied(Corner *corner, Color color) {
  if (color == NoColor) {
    return corner->occupancy == Village || corner->occupancy == City;
  } else {
    return (corner->occupancy == Village || corner->occupancy == City) && corner->color == color;
  }
}

bool corner_village_available(Corner *corner) {
  if (corner_occupied(corner, NoColor)) {
    return false;
  } else {
    bool valid = true;
    for (auto & street : corner->streets) {
      if (street != nullptr) {
        if (corner_occupied(street->corners[0], NoColor)) {
          valid = false;
        }
        if (corner_occupied(street->corners[1], NoColor)) {
          valid = false;
        }
      }
    }
  }
  return true;
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

Move *Player::update_available_moves(TurnType turn_type, Player *players[4]) {
  int current_move_id = 0;
  // Villages
  if (turn_type == openingTurnVillage || turn_type == normalTurn) {
    for (int corner_i = 0; corner_i < amount_of_corners; ++corner_i) {
      if (corner_village_available(&board->corner_array[corner_i])
       && resources_for_village()) {
        if (current_move_id >= max_available_moves) {
          printf("Ran out of available moves!");
          return available_moves;
        }
        available_moves[current_move_id] = Move();
        available_moves[current_move_id].move_type = buildVillage;
        available_moves[current_move_id].index = corner_i;
        ++current_move_id;
      }
    }
  }

  // Cities
  if (turn_type == normalTurn) {
    for (int corner_i = 0; corner_i < amount_of_corners; ++corner_i) {
      if (board->corner_array[corner_i].occupancy == Village
       && board->corner_array[corner_i].color == player_color
       && resources_for_city()) {
        if (current_move_id >= max_available_moves) {
          printf("Ran out of available moves!");
          return available_moves;
        }
        available_moves[current_move_id] = Move();
        available_moves[current_move_id].move_type = buildCity;
        available_moves[current_move_id].index = corner_i;
        ++current_move_id;
      }
    }
  }

  // Streets
  if (turn_type == openingTurnStreet || turn_type == normalTurn) {
    for (int street_i = 0; street_i < amount_of_streets; ++street_i) {
      if (board->street_array[street_i].color == NoColor
       && (corner_occupied(board->street_array[street_i].corners[0], player_color)
        || corner_occupied(board->street_array[street_i].corners[1], player_color))
       && resources_for_street()) {
        if (current_move_id >= max_available_moves) {
          printf("Ran out of available moves!");
          return available_moves;
        }
        available_moves[current_move_id] = Move();
        available_moves[current_move_id].move_type = buildStreet;
        available_moves[current_move_id].index = street_i;
        ++current_move_id;
      }
    }
  }

  // Development Card
  // TODO implement development cards

  // Trading


  // Exchanging
  if (turn_type == normalTurn) {

  }


  // Robber
  if (turn_type == robberTurn) {
    for (int tile_i = 0; tile_i < amount_of_tiles; ++tile_i) {
      if (!board->tile_array[tile_i].robber) {
        if (current_move_id >= max_available_moves) {
          printf("Ran out of available moves!");
          return available_moves;
        }
        available_moves[current_move_id] = Move();
        available_moves[current_move_id].move_type = moveRobber;
        available_moves[current_move_id].index = tile_i;
        ++current_move_id;
      }
    }
  }

  // End Turn
  if (turn_type == normalTurn) {
    if (current_move_id >= max_available_moves) {
      printf("Ran out of available moves!");
      return available_moves;
    }
    available_moves[current_move_id] = Move();
    available_moves[current_move_id].move_type = endTurn;
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
