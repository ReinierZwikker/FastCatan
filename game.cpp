//
// Created by reini on 25/04/2024.
//

#include <stdexcept>

#include "game.h"


Game::Game(int num_players) {
  if (num_players < 0 or num_players > 4) {
    throw std::invalid_argument("Maximum amount of players is four!");
  }

  Game::num_players = num_players;
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    Game::players[player_i] = Player();
  }

}

void Game::start_round() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    Game::players[player_i].start_round();
  }
}

void Game::step_round() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    Game::players[player_i].step_round();
  }
}
