#include <stdexcept>
#include <random>
#include <iostream>
#include <pthread.h>

#include "game.h"


Game::Game(int num_players) : gen(42), dice(1, 6), card(0, 4) {
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
  shuffle_development_cards();

  longest_road_player = nullptr;
  most_knights_player = nullptr;

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
    printf("Checking Move [%i]\n", move.type);
  }

  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (print) {
      printf("Move [%i]\n", available_moves[move_i].type);
    }

    if (available_moves[move_i].type == MoveType::NoMove) {
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
    printf("Move [%i] not available\nIndex: %i\nPlayer: %i\n", chosen_move.type, chosen_move.index, current_player_id + 1);
    printf("Information: %s\n", info.c_str());
    if (move.type == MoveType::Exchange) {
      printf("Exchange %i %s for %i %s\n", chosen_move.tx_amount, card_names[card_index(chosen_move.tx_card)].c_str(),
             chosen_move.rx_amount, card_names[card_index(chosen_move.rx_card)].c_str());
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
    if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
    if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
      unavailable_move(chosen_move, "first village");
    }
    current_player->place_village(chosen_move.index);

    // let player select first street
    current_player->set_cards(1, 1, 0, 0, 0);
    current_player->update_available_moves(openingTurnStreet, players, current_development_card);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
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
    if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
    if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
      unavailable_move(chosen_move, "second village");
    }
    current_player->place_village(chosen_move.index);

    // let player select second street
    current_player->set_cards(1, 1, 0, 0, 0);
    current_player->update_available_moves(openingTurnStreet, players, current_development_card);
    chosen_move = current_player->agent->get_move(&board, current_player->cards);
    if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
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
        if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
        move_robber(chosen_move.index);
      } else { give_cards(dice_roll); }

      for (int move_i = 0; move_i < moves_per_turn; ++move_i) {
        current_player->update_available_moves(normalTurn, players, current_development_card);
        chosen_move = current_player->agent->get_move(&board, current_player->cards);
        if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }

        if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
          unavailable_move(chosen_move, "normal turn");
        }

        // perform chosen move
        Move move;
        switch (chosen_move.type) {
          case MoveType::buildStreet:
            current_player->place_street(chosen_move.index);
            check_longest_road();
            break;
          case MoveType::buildVillage:
            current_player->place_village(chosen_move.index);
            check_longest_road_interrupt();
            break;
          case MoveType::buildCity:
            current_player->place_city(chosen_move.index);
            break;
          case MoveType::buyDevelopment:
            // Get the next development card of the deck
            if (current_development_card < amount_of_development_cards) {
              current_player->buy_development(development_cards[current_development_card]);
              ++current_development_card;
            }
            else {
              throw std::invalid_argument("Development Cards out of range");
            }
            break;
          case MoveType::playDevelopment:
            switch(current_player->development_cards[chosen_move.index].type) {
              case Knight:
                check_knights_played();

                current_player->update_available_moves(robberTurn, players, current_development_card);
                move = current_player->agent->get_move(&board, current_player->cards);
                if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }

                if (!move_in_available_moves(move, current_player->available_moves)) {
                  unavailable_move(move, "dev Knight turn");
                }

                move_robber(move.index);

                break;

              case Monopoly:
                // Chose a card to steal
                current_player->update_available_moves(devTurnYearOfPlenty, players, current_development_card);
                move = current_player->agent->get_move(&board, current_player->cards);
                if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }

                if (!move_in_available_moves(move, current_player->available_moves)) {
                  unavailable_move(move, "dev Year of Plenty turn");
                }

                // Steal the chosen card type from the other players
                for (int i = 0; i < num_players; ++i) {
                  current_player->cards[move.index] += players[i]->cards[move.index];
                  players[i]->cards[move.index] = 0;
                }

                break;

              case YearOfPlenty:
                // Let the player chose two free cards
                for (int card_i = 0; card_i < 2; ++card_i) {
                  current_player->update_available_moves(devTurnYearOfPlenty, players, current_development_card);
                  move = current_player->agent->get_move(&board, current_player->cards);
                  if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }

                  if (!move_in_available_moves(move, current_player->available_moves)) {
                    unavailable_move(move, "dev Year of Plenty turn");
                  }

                  current_player->add_cards(index_card(move.index), 1);
                }
                break;

              case RoadBuilding:
                // Let the player place two streets
                for (int street_i = 0; street_i < 2; ++street_i) {
                  current_player->add_cards(CardType::Brick, 1);
                  current_player->add_cards(CardType::Lumber, 1);
                  current_player->update_available_moves(devTurnStreet, players, current_development_card);
                  move = current_player->agent->get_move(&board, current_player->cards);
                  if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }

                  if (move.type != MoveType::NoMove) {
                    if (!move_in_available_moves(move, current_player->available_moves)) {
                      unavailable_move(move, "dev Road Building turn");
                    }

                    current_player->place_street(chosen_move.index);
                  }
                }
                check_longest_road();

                break;
            }
            current_player->play_development(chosen_move.index);

            break;
          case MoveType::Trade:
            // TODO Implement trading
            break;
          case MoveType::Exchange:
            current_player->remove_cards(chosen_move.tx_card, 4);
            current_player->add_cards(chosen_move.rx_card, 1);
            break;
          case MoveType::moveRobber:
            board.current_robber_tile->robber = false;
            board.current_robber_tile = &board.tile_array[chosen_move.index];
            board.current_robber_tile->robber = true;
            // TODO Implement robber movement
            // Only possible if development cards are implemented
            // TODO Implement card stealing?
            break;
          case MoveType::NoMove:
            throw std::invalid_argument("No Move is never a valid move!");
            break;
          case MoveType::endTurn:
            move_i = moves_per_turn;
            break;
        }

        // Only win if victory points >= 10 on your turn
        if (players[player_i]->check_victory_points() >= 10) {
          game_state = GameFinished;
          game_winner = index_color(player_i);
          break;
        }
      }

      current_player->activate_development_cards();
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

void Game::reset() {
  game_winner = Color::NoColor;
  current_round = 0;
  board.Reset();

  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    delete(players[player_i]->agent);
    delete(players[player_i]);
  }
  current_player = nullptr;
  longest_road_player = nullptr;
  most_knights_player = nullptr;
  add_players();

  shuffle_development_cards();
  current_development_card = 0;

  move_id = 0;

  gen.seed(seed); // Reseed
  game_state = ReadyToStart;
}

void Game::set_seed(unsigned int input_seed) {
  seed = input_seed;
  board.seed = input_seed;
}

int Game::roll_dice() {
  die_1 = dice(gen);
  die_2 = dice(gen);

  // printf("Rolled dice: %d + %d = %d\n", die_1, die_2, die_1 + die_2);

  return die_1 + die_2;
}

/*
 * Check which player has the longest trade route to give that player 2 VPs
 */
void Game::check_longest_road() {
  if (current_player->longest_route > 4) {
    if (longest_road_player == nullptr) {
      longest_road_player = current_player;
      longest_road_player->road_leader = true;
    }
    else if (current_player->longest_route > longest_road_player->longest_route) {
      longest_road_player->road_leader = false;
      longest_road_player = current_player;
      longest_road_player->road_leader = true;
    }
  }
}

void Game::check_longest_road_interrupt() {
  unsigned int longest_route = 0;
  Player* local_longest_road_player = nullptr;
  bool tie = false;

  for (Player *local_player : players) {
    if (local_player->longest_route > longest_route) {
      longest_route = local_player->longest_route;
      local_longest_road_player = local_player;
      tie = false;  // Reset the player tie
    }
    else if (local_player->longest_route == longest_route) {
      tie = true;
    }
  }

  if (local_longest_road_player != longest_road_player) {
    if (!tie && local_longest_road_player->longest_route > 4) {
      if (longest_road_player != nullptr) {
        longest_road_player->road_leader = false;
      }
      longest_road_player = local_longest_road_player;
      longest_road_player->road_leader = true;
    }
    else {
      longest_road_player = nullptr;
    }
  }

}

/*
 * Check which player has played the most knights to give that player 2 VPs
 */
void Game::check_knights_played() {
  if (current_player->played_knight_cards > 2) {
    if (most_knights_player == nullptr) {
      most_knights_player = current_player;
      most_knights_player->knight_leader = true;
    }
    else if (current_player->played_knight_cards > most_knights_player->played_knight_cards) {
      most_knights_player->knight_leader = false;
      most_knights_player = current_player;
      most_knights_player->knight_leader = true;
    }
  }
}

/*
 * Sets all tiles to no robbers and then sets the inserted tile_index to have a robber
 */
void Game::move_robber(int tile_index) {
  board.current_robber_tile->robber = false;
  board.current_robber_tile = &board.tile_array[tile_index];
  board.current_robber_tile->robber = true;
}

void Game::shuffle_development_cards() {
  auto rng = std::default_random_engine {seed};
  std::shuffle(development_cards, development_cards + amount_of_development_cards, rng);
}

/*
 * Give cards to players for each tile of the rolled number, or for all tiles if rolled_number = -1
 */
void Game::give_cards(int rolled_number) {
  // TODO Add check for bank reserves?
  for (Tile tile : board.tile_array) {
    if ((tile.number_token == rolled_number && !tile.robber) || rolled_number == -1) {
      for (Corner *corner : tile.corners) {
        if (corner->color != Color::NoColor) {
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

void Game::add_move_to_log(Move move) {

  if (log->move_file && (log->type == MoveLog || log->type == BothLogs)) {
    log->moves[log->writes] = move;
    ++log->writes;
  }
}

