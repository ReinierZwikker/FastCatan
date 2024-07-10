#include <stdexcept>

#include "game.h"


Game::Game(int num_players) {
  if (num_players < 0 or num_players > 6) {
    throw std::invalid_argument("Maximum amount of players is six!");
  }

  Game::num_players = num_players;
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    Game::players[player_i].activated = true;
  }

}

void Game::start_game() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    bool valid_choice = false;
    while (not valid_choice) {
      // let player select first town
      players[player_i];
      valid_choice = board.CheckValidity();
    }
  }

  for (int player_i = Game::num_players; player_i >= 0; player_i--) {
    // let player select second town

  }

}

void Game::step_round() {

}

void Game::give_cards(int rolled_number) {

}
