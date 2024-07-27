#include <stdexcept>

#include "game.h"


Game::Game(int num_players) {
  if (num_players < 0 or num_players > 4) {
    throw std::invalid_argument("Maximum amount of players is four!");
  }

  Game::num_players = num_players;
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    players[player_i] = Player(&board, index_color(player_i));
    players[player_i].activated = true;
  }

}

void Game::start_game() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    bool valid_choice = false;
    while (not valid_choice) {
      // let player select first town
      Move move = players[player_i].first_town();
      valid_choice = CheckValidity(move, openingMove);
      if (valid_choice) {
        // set choice on board
      }
    }
  }

  for (int player_i = Game::num_players; player_i >= 0; player_i--) {
    bool valid_choice = false;
    while (not valid_choice) {
      // let player select second town
      Move move = players[player_i].second_town();
      valid_choice = CheckValidity(move, openingMove);
      if (valid_choice) {
        // set choice on board
      }
    }
  }

}

void Game::step_round() {
  int rolled_number = roll_dice();

  give_cards(rolled_number);

  for (Player player : players) {
    if (player.activated) {
      bool valid_choice = false;
      while (not valid_choice) {
        Move move = player.do_move();
        valid_choice = CheckValidity(move);
        if (valid_choice) {
          // set choice on board
        }
      }
    }
  }

}

int Game::roll_dice() {
  // TODO generate random roll with 2d6 probabilities
  return game_seed * 1;
}

void Game::give_cards(int rolled_number) {
  // TODO Add check for bank reserves
  for (Tile *tile : board.tiles) { if (tile->number_token == rolled_number) {
    for (Corner *corner : tile->corners) { if (corner->color != NoColor) {
      if (corner->occupancy == Village) {
        players[color_index(corner->color)].cards[card_index(tile2card(tile->type))]++;
      } else if (corner->occupancy == City) {
        players[color_index(corner->color)].cards[card_index(tile2card(tile->type))] += 2;
      }
    } }
  } }
}

bool Game::CheckValidity() {
  // TODO
  // Check if all villages, cities, and streets are valid



  return false;
}

bool Game::CheckValidity(Move move) {
  // TODO
  // Check if proposed move is valid



  return false;
}

bool Game::CheckValidity(Move move, MoveType move_type) {
  // TODO
  // Check if proposed move of set type is valid
  if (move.move_type != move_type) {
    return false;
  }



  return false;
}