#include <stdexcept>
#include <iostream>

#include "game.h"


Game::Game(int num_players) {
  if (num_players < 0 or num_players > 4) {
    throw std::invalid_argument("Maximum amount of players is four!");
  }

  Game::num_players = num_players;
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    players[player_i] = new Player(&board, index_color(player_i));
    auto *new_agent = new HumanPlayer(players[player_i]);
    players[player_i]->agent = new_agent;
    players[player_i]->activated = true;
  }

  game_state = GameStates::ReadyToStart;
}

Game::~Game() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    free(players[player_i]->agent);
  }
}

bool move_in_available_moves(Move move, Move *available_moves) {
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (available_moves[move_i].move_type == NoMove) {
      return false;
    }

    if (move == available_moves[move_i]) {
      return true;
    }
  }
  return false;
}

void Game::start_game() {
  game_state = GameStates::Starting;

  Move chosen_move;
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    current_player = players[player_i];

    // std::unique_lock<std::mutex> lock(human_turn);

    // game_state = GameStates::WaitingForPlayer;

    // cv.wait(lock, [this] { return input_received; });

    // game_state = GameStates::Starting;

    // let player select first town
    current_player->set_cards(1, 1, 0, 1, 1);
    current_player->update_available_moves(openingTurnVillage, players);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
      throw std::invalid_argument("Not an available move!");
    }
    current_player->place_village(chosen_move.index);

    // let player select first street
    current_player->set_cards(1, 1, 0, 0, 0);
    current_player->update_available_moves(openingTurnStreet, players);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
      throw std::invalid_argument("Not an available move!");
    }
    current_player->place_street(chosen_move.index);
  }

  for (int player_i = Game::num_players; player_i >= 0; player_i--) {

    Player *current_player = players[player_i];

    // let player select second town
    current_player->set_cards(1, 1, 0, 1, 1);
    current_player->update_available_moves(openingTurnVillage, players);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (!move_in_available_moves(chosen_move, current_player->available_moves))
    { throw std::invalid_argument("Not an available move!"); }
    current_player->place_village(chosen_move.index);

    // let player select second street
    current_player->set_cards(1, 1, 0, 0, 0);
    current_player->update_available_moves(openingTurnStreet, players);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (!move_in_available_moves(chosen_move, current_player->available_moves))
    { throw std::invalid_argument("Not an available move!"); }
    current_player->place_street(chosen_move.index);
  }

  // Give starting cards, as if all numbers are rolled
  give_cards(-1);

}

void Game::human_input_received() {
  std::lock_guard<std::mutex> lock(human_turn);
  input_received = true;
  cv.notify_one();
}

void Game::step_round() {
  Move chosen_move;

  for (int player_i = 0; player_i < Game::num_players; player_i++) {

    Player *current_player = players[player_i];

    if (players[player_i]->activated) {
      int dice_roll = roll_dice();

      if (dice_roll == 7) {
        // do robber turn
        current_player->update_available_moves(robberTurn, players);
        chosen_move = current_player->agent->get_move(&board, current_player->cards);
        // perform chosen move
      } else { give_cards(dice_roll); }

      for (int move_i = 0; move_i < moves_per_turn; ++move_i) {
        current_player->update_available_moves(normalTurn, players);
        chosen_move = current_player->agent->get_move(&board, current_player->cards);


        if (!move_in_available_moves(chosen_move, current_player->available_moves))
          { throw std::invalid_argument("Not an available move!"); }

        // perform chosen move
        switch (chosen_move.move_type) {
          case buildStreet:
            current_player->place_street(chosen_move.index);
            break;
          case buildVillage:
            current_player->place_village(chosen_move.index);
            break;
          case buildCity:
            current_player->place_city(chosen_move.index);
            break;
          case buyDevelopment:
            // TODO Implement development cards
            break;
          case Trade:
            // TODO Implement trading
            break;
          case Exchange:
            current_player->cards[card_index(chosen_move.tx_card)] -= 4;
            current_player->cards[card_index(chosen_move.rx_card)]++;
            break;
          case moveRobber:
            // TODO Implement robber movement
            // Only possible if development cards are implemented
            // TODO make robber position changeable
            break;
          case NoMove:
            throw std::invalid_argument("No Move is never a valid move!");
            break;
          case endTurn:
            move_i = moves_per_turn;
            break;
        }
      }
    }
  }
}


int Game::roll_dice() {
  // TODO generate random roll with 2d6 probabilities
  return game_seed * 1;
}


/*
 * Give cards to players for each tile of the rolled number, or for all tiles if rolled_number = -1
 */
void Game::give_cards(int rolled_number) {
  // TODO Add check for bank reserves
  for (Tile *tile : board.tiles) { if (tile->number_token == rolled_number || rolled_number == -1) {
    for (Corner *corner : tile->corners) { if (corner->color != NoColor) {
      if (corner->occupancy == Village) {
        players[color_index(corner->color)]->cards[card_index(tile2card(tile->type))]++;
      } else if (corner->occupancy == City) {
        players[color_index(corner->color)]->cards[card_index(tile2card(tile->type))] += 2;
      }
    } }
  } }
}
