#include <iostream>

#include "player.h"


Player::Player(Board *global_board, Color player_color) {

  board = global_board;
  player_color = player_color;

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

Move Player::first_town(...) {
  Move move = Move();
  move.move_type = openingMove;

  // agent call
  //

  //move.corner = ;
  //move.street = ;


  return move;
}

Move Player::second_town(...) {
  Move move = Move();
  move.move_type = openingMove;

  // agent call
  //

  //move.corner = ;
  //move.street = ;


  return move;
}

Move Player::do_move(...) {
  Move move = Move();
  move.move_type = NoMove;

  // agent call
  //

  //move.corner = ;
  //move.street = ;


  return move;
}
