#include <stdexcept>

#include "game.h"


Game::Game(int num_players) {
  if (num_players < 0 or num_players > 4) {
    throw std::invalid_argument("Maximum amount of players is four!");
  }

  Game::num_players = num_players;
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    players[player_i] = Player(&board, index_color(player_i));
    players[player_i].agent = HumanPlayer();
    players[player_i].activated = true;
  }

}

bool move_in_available_moves(Move move, Move *available_moves) {
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {

  }
}

void Game::start_game() {
  Move chosen_move;
  for (int player_i = 0; player_i < Game::num_players; player_i++) {

    // let player select first town
    players[player_i].set_cards(1, 1, 0, 1, 1);
    players[player_i].update_available_moves(openingTurnFirstVillage);
    chosen_move = players[player_i].agent.get_move(&board, players[player_i].cards);
    players[player_i].place_village(chosen_move.index);

    // let player select first street
    players[player_i].set_cards(1, 1, 0, 0, 0);
    players[player_i].update_available_moves(openingTurnFirstStreet);
    chosen_move = players[player_i].agent.get_move(&board, players[player_i].cards);
    players[player_i].place_street(chosen_move.index);
  }

  for (int player_i = Game::num_players; player_i >= 0; player_i--) {

    // let player select second town
    players[player_i].set_cards(1, 1, 0, 1, 1);
    players[player_i].update_available_moves(openingTurnSecondVillage);
    chosen_move = players[player_i].agent.get_move(&board, players[player_i].cards);
    players[player_i].place_village(chosen_move.index);

    // let player select second street
    players[player_i].set_cards(1, 1, 0, 0, 0);
    players[player_i].update_available_moves(openingTurnSecondStreet);
    chosen_move = players[player_i].agent.get_move(&board, players[player_i].cards);
    players[player_i].place_street(chosen_move.index);
  }
}

void Game::step_round() {
  Move chosen_move;

  for (int player_i = 0; player_i < Game::num_players; player_i++) {



    if (players[player_i].activated) {
      give_cards(roll_dice());

      players[player_i].update_available_moves(normalTurn);
      chosen_move = players[player_i].agent.get_move(&board, players[player_i].cards);
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

