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
  if (board->corners[corner_id]->color == NoColor &&
      board->corners[corner_id]->occupancy == EmptyCorner &&
      resources_left[1] > 0) {
    board->street_array[corner_id].color = player_color;
    resources_left[1]--; // Remove one village from pool
    return 0;
  } else {
    return 1;
  }
}

int Player::place_city(int corner_id) {
  if (board->corners[corner_id]->color == player_color &&
      board->corners[corner_id]->occupancy == Village &&
      resources_left[2] > 0) {
    board->street_array[corner_id].color = player_color;
    resources_left[2]--; // Remove one city from pool
    resources_left[1]++; // Return one village to pool
    return 0;
  } else {
    return 1;
  }
}

Move *Player::update_available_moves(TurnType turn_type) {
  available_moves[0].move_type = endTurn;
  return available_moves;
}

void Player::set_cards(int brick, int lumber, int ore, int grain, int wool) {
  cards[0] = brick;
  cards[1] = lumber;
  cards[2] = ore;
  cards[3] = grain;
  cards[4] = wool;
}
