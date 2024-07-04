#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

#include "board.h"

bool compare_number_tokens(int number_token_1, int number_token_2) {
  bool number_token_1_check = false;
  bool number_token_2_check = false;

  if (number_token_1 == 6 || number_token_1 == 8) {
    number_token_1_check = true;
  }
  if (number_token_2 == 6 || number_token_2 == 8) {
    number_token_2_check = true;
  }
  if (number_token_1_check && number_token_2_check)
  {
    return true;
  }
  else
  {
    return false;
  }
}


Board::Board() {

  CalculateTileDifference();
  InitializeTilesAndTokens();

  // ## Create the map ##
  bool shuffling_map = true;
  while (shuffling_map) {
    ShuffleTilesAndTokens();
    AddNumberTokensToTiles();

    // ## Check number tokens ##
    if (CheckNumberTokens()) {
      shuffling_map = false;
    }
  }

  RewriteBoardLayout();
  LinkCornersAndStreetsToTiles();

  AddHarbors();

  // Temporary Check
  for (int tile_i = 0; tile_i < amount_of_tiles; tile_i++) {
    std::cout << "Tile " << tile_i << " : " << tile_names[tiles[tile_i].type] << " "
              << tiles[tile_i].number_token << std::endl;
  }
}

/*
 * Calculates the difference in column size between two rows.
 * This is repeated for every row of hexagons on the board and
 * added to the tile_diff array.
 */
void Board::CalculateTileDifference() {
  for (int row = 0; row < board_rows - 1; row++) {
    tile_diff[row] = tiles_in_row[row] - tiles_in_row[row + 1];
  }
}

/*
 * Initializes the tiles array with all possible tiles and the
 * number_tokens array with all possible number tokens.
 */
void Board::InitializeTilesAndTokens() {
  // ## Initialize tiles ##
  int current_tile_i = 0;
  for (int tile_type_i = 0; tile_type_i < 6; tile_type_i++) {
    for (int tile_i = 0; tile_i < max_terrain_tiles[tile_type_i]; tile_i++) {
      // Construct a new tile
      Tile current_tile = {};

      // Set initial values
      current_tile.type = tile_order[tile_type_i];

      // Robber starts on the Desert Tile
      if (current_tile.type == Desert) {
        current_tile.robber = true;
      } else {
        current_tile.robber = false;
      }

      // Append to tiles
      tiles[current_tile_i] = current_tile;
      current_tile_i++;
    }
  }
  if (current_tile_i != amount_of_tiles) { throw std::invalid_argument("Tiles do not add up!"); }

  // ## Initialize number tokens ##
  int current_number_token_i = 0;
  for (int number_token_i = 0; number_token_i < 11; number_token_i++) {
    for (int i = 0; i < max_number_tokens[number_token_i]; i++) {
      number_tokens[current_number_token_i] = number_token_i + 2;

      current_number_token_i++;
    }
  }
  if (current_number_token_i != amount_of_tokens) { throw std::invalid_argument("Number tokens do not add up!"); }
}

/*
 * Randomly shuffles the tiles and tokens using the default random engine.
 */
void Board::ShuffleTilesAndTokens() {
  auto random_seed = std::random_device {};
  auto rng = std::default_random_engine {random_seed()};
  std::shuffle(tiles, tiles + amount_of_tiles, rng);
  std::shuffle(number_tokens, number_tokens + amount_of_tokens, rng);
}

/*
 * Adds the number tokens to the tile structs
 */
void Board::AddNumberTokensToTiles() {
  int current_token = 0;
  for (auto & tile : tiles) {
    if (tile.type != tile_type::Desert) {
      tile.number_token = number_tokens[current_token];

      current_token++;
    }
  }
}

/*
 * Checks if the number tokens follow the rules, namely:
 * - Adjacent number tokens can not be two 6's
 * - Adjacent number tokens can not be two 8's
 * - Adjacent number tokens can not be a 6 and an 8
 */
bool Board::CheckNumberTokens() {
  // ## Check number tokens ##
  int current_column = 0;
  int current_row = 0;
  int difference = 0;
  int previous_difference = 0;
  bool mistake_found = false;

  for (int tile_i = 0; tile_i < amount_of_tiles; tile_i++) {
    if (current_column + 1 > tiles_in_row[current_row]) {
      current_column = 0;
      current_row++;
    }

    if (tile_diff[current_row] == -1) {
      difference = 1;
    }
    else if (tile_diff[current_row] == 1){
      difference = 0;
    }
    else {}

    if (tile_diff[current_row - 1] == -1 && current_row != 0) {
      previous_difference = 0;
    }
    else if (tile_diff[current_row - 1] == 1 && current_row != 0){
      previous_difference = 1;
    }

    if (current_column + 1 != tiles_in_row[current_row]) {
      mistake_found = compare_number_tokens(tiles[tile_i].number_token, tiles[tile_i + 1].number_token);
      if (mistake_found) {
        if (show_number_token_debug) {
          std::cout << "1 Found mistake in number tokens between tile: " << tile_i << " and " << tile_i + 1
                    << std::endl;
        }
        return false;
      }
    }

    if (current_column != tiles_in_row[current_row]) {
      // Middle row
      if (current_row != 0 && current_row + 1 != board_rows) {
        if (previous_difference != 0 || current_column + 1 != tiles_in_row[current_row]) {
          int tile_id_top = tile_i - tiles_in_row[current_row - 1] + previous_difference;
          mistake_found = compare_number_tokens(tiles[tile_i].number_token, tiles[tile_id_top].number_token);
          if (mistake_found) {
            if (show_number_token_debug) {
              std::cout << "2 Found mistake in number tokens between tile: " << tile_i << " and " << tile_id_top
                        << std::endl;
            }
            return false;
          }
        }

        if (difference != 0 || current_column + 1 != tiles_in_row[current_row]) {
          int tile_id_bottom = tile_i + tiles_in_row[current_row] + difference;
          mistake_found = compare_number_tokens(tiles[tile_i].number_token, tiles[tile_id_bottom].number_token);
          if (mistake_found) {
            if (show_number_token_debug) {
              std::cout << "3 Found mistake in number tokens between tile: " << tile_i << " and " << tile_id_bottom
                        << std::endl;
            }
            return false;
          }
        }
      }
        // Top row
      else if (current_row == 0) {
        if (difference != 0 || current_column + 1 != tiles_in_row[current_row]) {
          int tile_id_bottom = tile_i + tiles_in_row[current_row] + difference;
          mistake_found = compare_number_tokens(tiles[tile_i].number_token, tiles[tile_id_bottom].number_token);
          if (mistake_found) {
            if (show_number_token_debug) {
              std::cout << "4 Found mistake in number tokens between tile: " << tile_i << " and " << tile_id_bottom
                        << std::endl;
            }
            return false;
          }
        }
      }
        // Bottom row
      else {
        if (previous_difference != 0 || current_column + 1 != tiles_in_row[current_row]) {
          int tile_id_top = tile_i - tiles_in_row[current_row - 1] + previous_difference;
          mistake_found = compare_number_tokens(tiles[tile_i].number_token, tiles[tile_id_top].number_token);
          if (mistake_found) {
            if (show_number_token_debug) {
              std::cout << "5 Found mistake in number tokens between tile: " << tile_i << " and " << tile_id_top
                        << std::endl;
            }
            return false;
          }
        }
      }
    }

    current_column++;
  }

  return true;
}

/*
 * Write the board layout to a more usable form for further calculations.
 */
void Board::RewriteBoardLayout() {
  for (int row = 0; row < board_rows; row++) {
    if (row < board_rows - 1){
      tile_diff[row] = tiles_in_row[row] - tiles_in_row[row + 1];

      if (tile_diff[row] == 1) {
        row_decrease[row + 1] = 1;
      }
      else if (tile_diff[row] == -1) {
        row_decrease[row + 1] = 0;
      }
      else {
        throw std::invalid_argument("Map layout not supported!");
      }
    }
    else {
      row_decrease[row + 1] = 1;
    }

    previous_rows[row + 1] = previous_rows[row] + 2 * tiles_in_row[row] + 1 + 2 * row_decrease[row];
  }
  previous_rows[0] = 0;  // I don't know why this has to be there, but it breaks without it
}

/*
 * Link the corners and streets to the respective tile
 * and repeat this for all tiles.
 */
void Board::LinkCornersAndStreetsToTiles() {
  int current_column = 0;
  int current_row = 0;

  for (auto & tile : tiles) {
    // Check if the next row is reached
    if (current_column + 1 > tiles_in_row[current_row]) {
      current_column = 0;
      current_row++;
    }

    // Top side of the tile
    for (int corner_i = 0; corner_i < 3; corner_i++) {
      int corner_id = corner_i + 2 * current_column + previous_rows[current_row] + row_decrease[current_row];
      tile.corners[corner_i] = &corners[corner_id];
      tile.streets[corner_i] = &streets[corner_id];
    }
    // Bottom side of the tile
    for (int corner_i = 0; corner_i < 3; corner_i++) {
      int corner_id = 3 - corner_i + 2 * current_column + previous_rows[current_row + 1] - row_decrease[current_row + 1];
      tile.corners[corner_i + 3] = &corners[corner_id];
      tile.streets[corner_i + 3] = &streets[corner_id];
    }

    current_column++;
  }
}

/*
 * Adds harbor types to pre-defined corners of selected tiles.
 */
void Board::AddHarbors() {
  for (auto harbor : harbors) {
    tiles[harbor.tile_id].corners[harbor.corner_1]->harbor = harbor.type;
    tiles[harbor.tile_id].corners[harbor.corner_2]->harbor = harbor.type;
  }
}

/*
 * Prints board to the console.
 * */
void Board::print_board() {
  char board_chars[884] = "               .         .         .               \n"
                          "          .         .         .         .          \n"
                          "              X00       X00       X00              \n"
                          "          .         .         .         .          \n"
                          "     .         .         .         .         .     \n"
                          "         X00       X00       X00       X00         \n"
                          "     .         .         .         .         .     \n"
                          ".         .         .         .         .         .\n"
                          "    X00       X00       X00       X00       X00    \n"
                          ".         .         .         .         .         .\n"
                          "     .         .         .         .         .     \n"
                          "         X00       X00       X00       X00         \n"
                          "     .         .         .         .         .     \n"
                          "          .         .         .         .          \n"
                          "              X00       X00       X00              \n"
                          "          .         .         .         .          \n"
                          "               .         .         .               ";

  int current_corner = 0;
  int current_tile = 0;

  for (int char_i = 0; char_i < 884; ++char_i) {
    switch (board_chars[char_i]) {
      case 'X':
        board_chars[char_i] = tile_shortnames[tiles[current_tile].type];
        if (tiles[current_tile].robber) {
          char_i++;
          board_chars[char_i] = '_';
          char_i++;
          board_chars[char_i] = 'R';
        } else if (tiles[current_tile].type == Desert) {
          char_i++;
          board_chars[char_i] = '_';
          char_i++;
          board_chars[char_i] = '_';
        } else if (tiles[current_tile].number_token < 10) {
          char_i += 2;
          board_chars[char_i] = '0' + tiles[current_tile].number_token;
        } else {
          char_i++;
          board_chars[char_i] = '1';
          char_i++;
          board_chars[char_i] = '0' + (tiles[current_tile].number_token - 10);
        }
        current_tile++;
        break;
      case '.':
        // TODO Check if village or city, fill in here
        break;
      // TODO draw streets
      default:
        break;
    }

  }


  printf("\n              ==   CURRENT BOARD   ==\n\n%s\n\n"
         "    D = Desert, H = Hills, F = Forest,\n"
         "    M = Mountains, f = Fields, P = Pasture\n", board_chars);
}
