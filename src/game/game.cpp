#include <stdexcept>
#include <random>
#include <iostream>
#include <pthread.h>

#include "game.h"


Game::Game(int num_players) {
  if (num_players < 0 or num_players > 4) {
    throw std::invalid_argument("Maximum amount of players is four!");
  }

  Game::num_players = num_players;
  add_players();

  // Initialize development cards
  int current_dev_card = 0;
  for (int dev_card_i = 0; dev_card_i < 5; ++dev_card_i) {
    for (int i = 0; i < max_development_cards[dev_card_i]; ++i) {
      development_cards[current_dev_card] = index_dev_card(dev_card_i);

      ++current_dev_card;
    }
  }

  game_state = GameStates::ReadyToStart;
}

Game::~Game() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    delete(players[player_i]->agent);
    delete(players[player_i]);
  }
}

void Game::add_players() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    players[player_i] = new Player(&board, index_color(player_i));
//    auto *new_agent = new GuiPlayer(players[player_i]);
    auto *new_agent = new RandomPlayer(players[player_i]);
    players[player_i]->agent = new_agent;
    players[player_i]->activated = true;
  }
}

bool move_in_available_moves(Move move, Move *available_moves, bool print = false) {
  if (print) {
    printf("Checking Move [%i]\n", move.move_type);
  }

  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (print) {
      printf("Move [%i]\n", available_moves[move_i].move_type);
    }

    if (available_moves[move_i].move_type == NoMove) {
      if (print) {
        printf("No Move\n");
      }
      return false;
    }

    if (move == available_moves[move_i]) {
      if (print) {
        printf("Move [%i] approved\n", move_i);
      }
      return true;
    }
  }
  return false;
}

void Game::unavailable_move(Move move, std::string info) {
  printf("\nMove Warning!\n");

  if (gui_controlled) {
    printf("Move [%i] not available\nIndex: %i\nPlayer: %i\n", chosen_move.move_type, chosen_move.index, current_player_id + 1);
    printf("Information: %s\n", info.c_str());
    if (move.move_type == Exchange) {
      printf("Exchange %i %s for %i %s\n", chosen_move.tx_amount, card_names[chosen_move.tx_card].c_str(),
             chosen_move.rx_amount, card_names[chosen_move.rx_card].c_str());
    }

    std::unique_lock<std::mutex> lock(move_lock);
    game_state = GameStates::UnavailableMove;
    cv.wait(lock, [this] { return move_lock_opened; });
  }
  else {
    throw std::invalid_argument("Not an available move!");
  }
}

void Game::start_game() {
  game_state = GameStates::SetupRound;

  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    current_player = players[player_i];
    current_player_id = player_i;

    // let player select first town
    current_player->set_cards(1, 1, 0, 1, 1);
    current_player->update_available_moves(openingTurnVillage, players, current_development_card);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
      unavailable_move(chosen_move, "first village");
    }
    current_player->place_village(chosen_move.index);

    // let player select first street
    current_player->set_cards(1, 1, 0, 0, 0);
    current_player->update_available_moves(openingTurnStreet, players, current_development_card);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
      unavailable_move(chosen_move, "first street");
    }
    current_player->place_street(chosen_move.index);
  }

  for (int player_i = Game::num_players-1; player_i >= 0; player_i--) {

    current_player = players[player_i];
    current_player_id = player_i;

    // let player select second town
    current_player->set_cards(1, 1, 0, 1, 1);
    current_player->update_available_moves(openingTurnVillage, players, current_development_card);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
      unavailable_move(chosen_move, "second village");
    }
    current_player->place_village(chosen_move.index);

    // let player select second street
    // TODO force player to select street adjacent to second village
    current_player->set_cards(1, 1, 0, 0, 0);
    current_player->update_available_moves(openingTurnStreet, players, current_development_card);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
      unavailable_move(chosen_move, "second street");
    }
    current_player->place_street(chosen_move.index);
  }

  // Give starting cards, as if all numbers are rolled
  give_cards(-1);

  game_state = GameStates::SetupRoundFinished;
}

void Game::human_input_received() {
  current_player->agent->unpause(gui_moves[current_player_id]);
}

void Game::step_round() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    current_player = players[player_i];
    current_player_id = player_i;

    if (players[player_i]->activated) {
      int dice_roll = roll_dice();

      if (dice_roll == 7) {
        // do robber turn
        for (int player_j = 0; player_j < Game::num_players; player_j++) {
          if (players[player_j]->get_total_amount_of_cards() > 7) {
            // TODO Allow player to choose which cards get discarded!
            std::mt19937 gen(randomDevice());
            std::uniform_int_distribution<> card(0, 5);
            for (int card_i = 0; card_i < players[player_j]->get_total_amount_of_cards() / 2; ++card_i) {
              bool chosen = false;
              while (!chosen) {
                int chosen_card = card(gen);
                if (players[player_j]->cards[chosen_card] > 0) {
                  players[player_j]->remove_cards(index_card(chosen_card), 1);
                  chosen = true;
                }
              }
            }
          }
        }
        current_player->update_available_moves(robberTurn, players, current_development_card);
        chosen_move = current_player->agent->get_move(&board, current_player->cards);
        // perform chosen move
      } else { give_cards(dice_roll); }

      for (int move_i = 0; move_i < moves_per_turn; ++move_i) {
        current_player->update_available_moves(normalTurn, players, current_development_card);
        chosen_move = current_player->agent->get_move(&board, current_player->cards);

        if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
          unavailable_move(chosen_move, "normal turn");
        }

        // perform chosen move
        switch (chosen_move.move_type) {
          case buildStreet:
            current_player->place_street(chosen_move.index);
            check_longest_trade_route();
            break;
          case buildVillage:
            current_player->place_village(chosen_move.index);
            break;
          case buildCity:
            current_player->place_city(chosen_move.index);
            break;
          case buyDevelopment:
            // TODO Implement development cards
            // Get the next development card of the deck
            current_player->add_development(development_cards[current_development_card]);
            ++current_development_card;
            break;
          case playDevelopment:
            current_player->play_development(chosen_move.index);
            check_knights_played();
            break;
          case Trade:
            // TODO Implement trading
            break;
          case Exchange:
            current_player->remove_cards(chosen_move.tx_card, 4);
            current_player->add_cards(chosen_move.rx_card, 1);
            break;
          case moveRobber:
            board.current_robber_tile->robber = false;
            board.current_robber_tile = &board.tile_array[chosen_move.index];
            board.current_robber_tile->robber = true;
            // TODO Implement robber movement
            // Only possible if development cards are implemented
            // TODO Implement card stealing?
            break;
          case NoMove:
            throw std::invalid_argument("No Move is never a valid move!");
            break;
          case endTurn:
            move_i = moves_per_turn;
            break;
        }

        // Only win if victory points >= 10 on your turn
        if (players[player_i]->check_victory_points() >= 10) {
          game_state = GameFinished;
          game_winner = index_color(player_i);
        }
      }

      current_player->played_development_card = false;
    }
  }

  ++current_round;
  if (game_state == PlayingRound) {
    game_state = RoundFinished;
  }
}

void Game::run_game() {
  this->start_game();

  while (game_state != GameFinished && current_round < max_rounds) {
    this->step_round();
  }

  game_state = GameFinished;
}

void Game::run_multiple_games() {
  while (keep_running) {
    run_game();
    this->reset();
    ++games_played;
  }
}

void Game::reset() {
  game_winner = NoColor;
  current_round = 0;
  board.Reset();

  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    delete(players[player_i]->agent);
    delete(players[player_i]);
  }
  current_player = nullptr;
  add_players();

  shuffle_development_cards();
  current_development_card = 0;

  game_state = ReadyToStart;
}

int Game::roll_dice() {
  std::mt19937 gen(randomDevice());
  std::uniform_int_distribution<> dice(1, 6);
  die_1 = dice(gen);
  die_2 = dice(gen);

  // printf("Rolled dice: %d + %d = %d\n", die_1, die_2, die_1 + die_2);

  return die_1 + die_2;
}

/*
 * Check which player has the longest trade route to give that player 2 VPs
 */
void Game::check_longest_trade_route() {
  if (current_player->longest_route > 2 && current_player->longest_route > longest_trade_route) {
    current_player->longest_trading_route = true;
    longest_trade_route = current_player->longest_route;

    for (int player_i = 0; player_i < num_players; ++player_i) {
      if (player_i != current_player_id) {
        players[player_i]->longest_trading_route = false;
      }
    }
  }
}

/*
 * Check which player has played the most knights to give that player 2 VPs
 */
void Game::check_knights_played() {
  if (current_player->played_knight_cards > 2 && current_player->played_knight_cards > most_played_knights) {
    current_player->most_knights_played = true;
    most_played_knights = current_player->played_knight_cards;

    for (int player_i = 0; player_i < num_players; ++player_i) {
      if (player_i != current_player_id) {
        players[player_i]->most_knights_played = false;
      }
    }
  }
}

void Game::shuffle_development_cards() {
  std::shuffle(development_cards, development_cards + amount_of_development_cards, randomDevice);
}

/*
 * Give cards to players for each tile of the rolled number, or for all tiles if rolled_number = -1
 */
void Game::give_cards(int rolled_number) {
  // TODO Add check for bank reserves?
  for (Tile tile : board.tile_array) {
    if ((tile.number_token == rolled_number && !tile.robber) || rolled_number == -1) {
      for (Corner *corner : tile.corners) {
        if (corner->color != NoColor) {
          if (corner->occupancy == Village) {
            players[color_index(corner->color)]->add_cards(tile2card(tile.type), 1);
            // std::cout << "Giving one " + card_name(tile2card(tile.type)) + " to player " + color_name(corner->color) + "." << std::endl;
          } else if (corner->occupancy == City) {
            players[color_index(corner->color)]->add_cards(tile2card(tile.type), 2);
            // std::cout << "Giving two " + card_name(tile2card(tile.type)) + " to player " + color_name(corner->color) + "." << std::endl;
          }
        }
      }
    }
  }
}
