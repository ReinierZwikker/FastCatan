#include <stdexcept>

#include "game.h"
#include "game/AIPlayer/random_player.h"


Game::Game(bool gui, int num_players, unsigned int input_seed) : gen(input_seed), dice(1, 6), card(0, 4) {
  if (num_players < 0 or num_players > 4) {
    throw std::invalid_argument("Maximum amount of players is four!");
  }

  Game::num_players = num_players;
  PlayerType player_type[4];
  if (gui) {
    for (int player_i = 0; player_i < num_players; ++player_i) {
      player_type[player_i] = PlayerType::guiPlayer;
    }
  }
  else {
    for (int player_i = 0; player_i < num_players; ++player_i) {
      player_type[player_i] = PlayerType::consolePlayer;
    }
  }
  add_players(player_type);

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

void Game::add_player(PlayerType player_type, int player_id) {
  if (players[player_id]->agent) {
    delete(players[player_id]->agent);
  }
  switch (player_type) {
    case PlayerType::consolePlayer: {
      auto *new_agent = new ConsolePlayer(players[player_id]);
      players[player_id]->agent = new_agent;
      assigned_players[player_id] = true;
      break;
    }
    case PlayerType::guiPlayer: {
      auto *new_agent = new GuiPlayer(players[player_id]);
      players[player_id]->agent = new_agent;
      assigned_players[player_id] = true;
      break;
    }
    case PlayerType::randomPlayer: {
      auto *new_agent = new RandomPlayer(players[player_id], rd());
      players[player_id]->agent = new_agent;
      assigned_players[player_id] = true;
      break;
    }
    case PlayerType::zwikPlayer: {
      throw std::invalid_argument("ZwikPlayer not available");
    }
    case PlayerType::beanPlayer: {
      throw std::invalid_argument("BeanPlayer not available");
    }
    case PlayerType::NoPlayer: {
      throw std::invalid_argument("A player must be selected");
    }
  }
  players[player_id]->activated = true;

}

void Game::add_players(PlayerType player_type[4]) {
//  for (int player_i = 0; player_i < Game::num_players; player_i++) {
//    if (players[player_i] != nullptr) {
//      delete(players[player_i]->agent);
//      delete(players[player_i]);
//    }
//  }
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    players[player_i] = new Player(&board, index_color(player_i));
    add_player(player_type[player_i], player_i);
  }
}

void Game::add_players(Player* new_players[4]) {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    if (new_players[player_i] != nullptr) {
      players[player_i] = new_players[player_i];
    }
  }
}

void Game::delete_players() {
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    if (players[player_i] != nullptr && assigned_players[player_i]) {
      assigned_players[player_i] = false;
      delete players[player_i]->agent;
      delete players[player_i];
    }
  }
}

bool move_in_available_moves(Move move, Move *available_moves, bool print = false) {
  if (print) {
    printf("Checking Move [%i]\n", (int)move.type);
  }

  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (print) {
      printf("Move [%i]\n", (int)available_moves[move_i].type);
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

void Game::unavailable_move(Move move, const std::string& info) {
  printf("\nMove Warning!\n");

  if (gui_controlled) {
    printf("Move [%s] not available\nIndex: %i\nPlayer: %s\n", move2string(chosen_move).c_str(),
           chosen_move.index, color_names[current_player_id].c_str());
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
  GameInfo game_info;

  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    current_player = players[player_i];
    current_player_id = player_i;

    if (current_player->activated) {
      // let player select first town
      current_player->set_cards(1, 1, 0, 1, 1);
      current_player->update_available_moves(TurnType::openingTurnVillage, players, current_development_card);

      game_info = {(uint8_t)current_development_card,
                   TurnType::openingTurnVillage, current_round};
      chosen_move = current_player->agent->get_move(&board, current_player->cards, game_info);

      if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
      if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
        unavailable_move(chosen_move, "first village");
      }
      current_player->place_village(chosen_move.index);

      // let player select first street
      current_player->set_cards(1, 1, 0, 0, 0);
      current_player->update_available_moves(TurnType::openingTurnStreet, players, current_development_card);

      game_info = {(uint8_t)current_development_card,
                   TurnType::openingTurnStreet, current_round};
      chosen_move = current_player->agent->get_move(&board, current_player->cards, game_info);

      if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
      if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
        unavailable_move(chosen_move, "first street");
      }
      current_player->place_street(chosen_move.index);
    }
  }

  for (int player_i = Game::num_players-1; player_i >= 0; player_i--) {

    current_player = players[player_i];
    current_player_id = player_i;

    if (current_player->activated) {
      // let player select second town
      current_player->set_cards(1, 1, 0, 1, 1);
      current_player->update_available_moves(TurnType::openingTurnVillage, players, current_development_card);

      game_info = {(uint8_t)current_development_card,
                   TurnType::openingTurnVillage, current_round};
      chosen_move = current_player->agent->get_move(&board, current_player->cards, game_info);

      if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
      if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
        unavailable_move(chosen_move, "second village");
      }
      current_player->place_village(chosen_move.index);

      // let player select second street
      current_player->set_cards(1, 1, 0, 0, 0);
      current_player->update_available_moves(TurnType::openingTurnStreet, players, current_development_card);

      game_info = {(uint8_t)current_development_card,
                   TurnType::openingTurnStreet, current_round};
      chosen_move = current_player->agent->get_move(&board, current_player->cards, game_info);

      if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
      if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
        unavailable_move(chosen_move, "second street");
      }
      current_player->place_street(chosen_move.index);
    }
  }

  // Give starting cards, as if all numbers are rolled
  give_cards(-1);

  game_state = GameStates::SetupRoundFinished;
}

void Game::human_input_received(Move move) {
  current_player->agent->unpause(move);
}

void Game::step_round() {

  GameInfo game_info{};
  for (int player_i = 0; player_i < Game::num_players; player_i++) {
    current_player = players[player_i];
    current_player_id = player_i;

    if (current_player->activated) {
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
        current_player->update_available_moves(TurnType::robberTurn, players, current_development_card);

        game_info = {(uint8_t)current_development_card,
                     TurnType::robberTurn, current_round};
        chosen_move = current_player->agent->get_move(&board, current_player->cards, game_info);

        if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }
        move_robber(chosen_move.index);
        if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
          unavailable_move(chosen_move, "robber turn");
        }

      } else { give_cards(dice_roll); }

      for (int move_i = 0; move_i < moves_per_turn; ++move_i) {
        current_player->update_available_moves(TurnType::normalTurn, players, current_development_card);

        game_info = {(uint8_t)current_development_card,
                     TurnType::normalTurn, current_round};
        chosen_move = current_player->agent->get_move(&board, current_player->cards, game_info);

        if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(chosen_move); }

        if (!move_in_available_moves(chosen_move, current_player->available_moves)) {
          unavailable_move(chosen_move, "normal turn");
        }

        // perform chosen move
        Move dev_move;
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
            switch(chosen_move.index) {
              case Knight:
                check_knights_played();

                current_player->update_available_moves(TurnType::devTurnKnight, players, current_development_card);

                game_info = {(uint8_t)current_development_card,
                             TurnType::normalTurn, current_round};
                dev_move = current_player->agent->get_move(&board, current_player->cards, game_info);

                if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(dev_move); }

                if (!move_in_available_moves(dev_move, current_player->available_moves)) {
                  unavailable_move(dev_move, "dev Knight turn");
                }

                move_robber(dev_move.index);
                --current_player->dev_cards[0];
                break;

              case Monopoly:
                // Chose a card to steal
                current_player->update_available_moves(TurnType::devTurnMonopoly, players, current_development_card);

                game_info = {(uint8_t)current_development_card,
                             TurnType::normalTurn, current_round};
                dev_move = current_player->agent->get_move(&board, current_player->cards, game_info);

                if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(dev_move); }

                if (!move_in_available_moves(dev_move, current_player->available_moves)) {
                  unavailable_move(dev_move, "dev Monopoly turn");
                }

                // Steal the chosen card type from the other players
                for (int i = 0; i < num_players; ++i) {
                  current_player->cards[dev_move.index] += players[i]->cards[dev_move.index];
                  players[i]->cards[dev_move.index] = 0;
                }
                --current_player->dev_cards[2];
                break;

              case YearOfPlenty:
                // Let the player chose two free cards
                for (int card_i = 0; card_i < 2; ++card_i) {
                  current_player->update_available_moves(TurnType::devTurnYearOfPlenty, players, current_development_card);

                  game_info = {(uint8_t)current_development_card,
                               TurnType::normalTurn, current_round};
                  dev_move = current_player->agent->get_move(&board, current_player->cards, game_info);

                  if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(dev_move); }

                  if (!move_in_available_moves(dev_move, current_player->available_moves)) {
                    unavailable_move(dev_move, "dev Year of Plenty turn");
                  }

                  current_player->add_cards(index_card(dev_move.index), 1);
                }
                --current_player->dev_cards[3];
                break;

              case RoadBuilding:
                // Let the player place two streets
                for (int street_i = 0; street_i < 2; ++street_i) {
                  current_player->add_cards(CardType::Brick, 1);
                  current_player->add_cards(CardType::Lumber, 1);
                  current_player->update_available_moves(TurnType::devTurnStreet, players, current_development_card);

                  game_info = {(uint8_t)current_development_card,
                               TurnType::normalTurn, current_round};
                  dev_move = current_player->agent->get_move(&board, current_player->cards, game_info);

                  if (log != nullptr && (log->type == MoveLog || log->type == BothLogs)) { add_move_to_log(dev_move); }

                  if (dev_move.type != MoveType::NoMove) {
                    if (!move_in_available_moves(dev_move, current_player->available_moves)) {
                      unavailable_move(dev_move, "dev Road Building turn");
                    }

                    current_player->place_street(chosen_move.index);
                  }
                }
                check_longest_road();
                --current_player->dev_cards[4];
                break;
              case VictoryPoint:
                throw std::invalid_argument("Cannot play a Victory Card");
              case None:
                throw std::invalid_argument("Tried playing a None Development Card");
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
          case MoveType::endTurn:
            move_i = moves_per_turn;
            break;
          default:
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
  gen.seed(seed); // Reseed

  game_winner = Color::NoColor;
  current_round = 0;
  board.Reset();
  board.Randomize();

  current_player = nullptr;
  longest_road_player = nullptr;
  most_knights_player = nullptr;

  delete_players();

  shuffle_development_cards();
  current_development_card = 0;

  move_id = 0;
  game_state = ReadyToStart;
}

void Game::reseed(unsigned int input_seed) {
  seed = input_seed;
  board.seed = input_seed;
}

int Game::roll_dice() {
  die_1 = dice(gen);
  die_2 = dice(gen);

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
  int current_dev_card = 0;
  for (int dev_card_i = 0; dev_card_i < 5; ++dev_card_i) {
    for (int i = 0; i < max_development_cards[dev_card_i]; ++i) {
      development_cards[current_dev_card] = index_dev_card(dev_card_i);

      ++current_dev_card;
    }
  }

  std::shuffle(development_cards, development_cards + amount_of_development_cards, gen);
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

void Game::add_move_to_log(Move move) const {
  if (log->move_file && (log->type == MoveLog || log->type == BothLogs)) {
    log->moves[log->writes] = move;
    ++log->writes;
  }
}
