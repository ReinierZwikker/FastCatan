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

void AddCorner2Street(Corner* corner, Street* street) {
  if (street->corners[0] == nullptr) {
    street->corners[0] = corner;
  }
  else if (street->corners[1] == nullptr) {
    street->corners[1] = corner;
  }
  else {
    std::cout << "Error encountered while linking corner [" << corner->id << "] to street [" << street->id <<
                 "], street [" << street->id << "] already has to corners linked (corner ["
                 << street->corners[0]->id << "] and corner[" << street->corners[1]->id << "]" << std::endl;
  }
}

Harbor::Harbor(int input_tile_id, int input_corner_1, int input_corner_2, HarborType harbor_type) {
  tile_id = input_tile_id;
  corners[0] = input_corner_1;
  corners[1] = input_corner_2;
  type = harbor_type;
}


Board::Board() {

  for (int corner = 0; corner < amount_of_corners; corner++) {
    corner_array[corner].id = corner;
  }

  for (int street = 0; street < amount_of_streets; street++) {
    street_array[street].id = street;
  }

  LinkParts();
  InitializeTilesAndTokens();
  Randomize();
  AddHarbors();

}

/*
 * Links all the memory addresses of corners and streets to their corresponding tile and
 * corners to streets and vise versa.
 */
void Board::LinkParts() {

  /*
   * Calculate difference in tiles in this row to the next row
   */
  for (int tile_row_i = 0; tile_row_i < tile_rows; tile_row_i++) {
    if (tile_row_i < tile_rows - 1) {
      tile_diff[tile_row_i] = tiles_in_row[tile_row_i + 1] - tiles_in_row[tile_row_i];
    }
    else {
      tile_diff[tile_row_i] = -1;
    }
  }

  /*
   * Link the memory addresses from the tile_array to the tile rows
   */
  int first_tile = 0;
  for (int tile_row_i = 0; tile_row_i < tile_rows; tile_row_i++) {
    tiles[tile_row_i] = &tile_array[first_tile];
    first_tile += tiles_in_row[tile_row_i];
  }

  /*
   * Link the memory addresses from the corner_array to the corner rows
   */
  int first_corner = 0;
  for (int corner_row_i = 0; corner_row_i < corner_rows; corner_row_i++) {
    corners[corner_row_i] = &corner_array[first_corner];
    first_corner += corners_in_row[corner_row_i];
  }

  /*
   * Link the memory addresses from the street_array to the street rows
   */
  int first_street = 0;
  for (int street_row_i = 0; street_row_i < street_rows; street_row_i++) {
    streets[street_row_i] = &street_array[first_street];
    first_street += streets_in_row[street_row_i];
  }

  /*
   * Link corners and streets to their respective tile
   */
  int shift_top, shift_bottom;
  for (int tile_row_i = 0; tile_row_i < tile_rows; tile_row_i++) {
    for (int tile_column_i = 0; tile_column_i < tiles_in_row[tile_row_i]; tile_column_i++) {
      int offset = abs(tile_diff[tile_row_i]);

      // Top
      shift_top = 2 * tile_column_i;
      if (tile_row_i > 0) {
        if (tile_diff[tile_row_i - 1] < 0) {
          shift_top += offset;
        }
      }

      // Bottom
      shift_bottom = 2 * tile_column_i + 2;
      if (tile_diff[tile_row_i] > 0) {
        shift_bottom += offset;
      }

      for (int i = 0; i < 3; i++) {
        // Corners
        tiles[tile_row_i][tile_column_i].corners[i] = &corners[tile_row_i][shift_top + i];
        tiles[tile_row_i][tile_column_i].corners[i + 3] = &corners[tile_row_i + 1][shift_bottom - i];

        // Streets
        if (i < 2) {
          tiles[tile_row_i][tile_column_i].streets[i] = &streets[2 * tile_row_i][shift_top + i];
          tiles[tile_row_i][tile_column_i].streets[i + 3] = &streets[2 * tile_row_i + 2][shift_bottom - 1 - i];
        }

      }
      // Streets at the right sides of the tiles
      tiles[tile_row_i][tile_column_i].streets[2] = &streets[2 * tile_row_i + 1][tile_column_i + 1];
      tiles[tile_row_i][tile_column_i].streets[5] = &streets[2 * tile_row_i + 1][tile_column_i];
    }
  }

  // Link streets to corners
  bool expanding = true;
  for (int row = 0; row < corner_rows; row++) {
    // Check if the corners are expanding
    if (row < corner_rows - 1 && corners_in_row[row] <= corners_in_row[row + 1]) {
      expanding = true;
    }
    else {
      expanding = false;
    }

    for (int column = 0; column < corners_in_row[row]; column++) {
      // Add only right
      if (column == 0) {
        corners[row][column].streets[2] = &streets[2 * row][column];
        AddCorner2Street(&corners[row][column], &streets[2 * row][column]);
      }
      // Add only left
      else if (column == corners_in_row[row] - 1) {
        corners[row][column].streets[0] = &streets[2 * row][column - 1];
        AddCorner2Street(&corners[row][column], &streets[2 * row][column - 1]);
      }
      // Add left and right
      else {
        corners[row][column].streets[2] = &streets[2 * row][column];
        AddCorner2Street(&corners[row][column], &streets[2 * row][column]);
        corners[row][column].streets[0] = &streets[2 * row][column - 1];
        AddCorner2Street(&corners[row][column], &streets[2 * row][column - 1]);
      }

      if (expanding) {
        // Add below
        if (column % 2 == 0) {
          corners[row][column].streets[1] = &streets[2 * row + 1][column / 2];
          AddCorner2Street(&corners[row][column], &streets[2 * row + 1][column / 2]);
        }
        // Add above
        else if (row != 0) {
          corners[row][column].streets[1] = &streets[2 * row - 1][(column - 1) / 2];
          AddCorner2Street(&corners[row][column], &streets[2 * row - 1][(column - 1) / 2]);
        }
      }
      else {
        // Add above
        if (column % 2 == 0) {
          corners[row][column].streets[1] = &streets[2 * row - 1][column / 2];
          AddCorner2Street(&corners[row][column], &streets[2 * row - 1][column / 2]);
        }
        // Add below
        else if (row < corner_rows - 1) {
          corners[row][column].streets[1] = &streets[2 * row + 1][(column - 1) / 2];
          AddCorner2Street(&corners[row][column], &streets[2 * row + 1][(column - 1) / 2]);
        }
      }
    }
  }

}

/*
 * Initializes the available_tiles and the
 * number_tokens array with all possible number tokens.
 */
void Board::InitializeTilesAndTokens() {
  // ## Initialize available_tiles ##
  int current_tile_type_i = 0;
  for (int tile_type_i = 0; tile_type_i < 6; tile_type_i++) {
    for (int i = 0; i < max_terrain_tiles[tile_type_i]; i++) {
      available_tiles[current_tile_type_i] = static_cast<TileType>(tile_type_i);

      current_tile_type_i++;
    }
  }

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
 * Randomly shuffles the tile_array and tokens using the default random engine.
 */
void Board::ShuffleTilesAndTokens() {
  auto random_seed = std::random_device {};
  auto rng = std::default_random_engine {random_seed()};
  std::shuffle(available_tiles, available_tiles + amount_of_tiles, rng);
  std::shuffle(number_tokens, number_tokens + amount_of_tokens, rng);
}

/*
 * Adds the number tokens to the tile structs
 */
void Board::AddTileTypeAndNumberTokensToTiles() {
  int current_tile = 0;
  for (int tile_i = 0; tile_i < amount_of_tiles; tile_i++) {
    tile_array[tile_i].type = available_tiles[tile_i];
  }

  int current_token = 0;
  for (auto & tile : tile_array) {
    if (tile.type != TileType::Desert) {
      tile.number_token = number_tokens[current_token];

      current_token++;
    }
    else {
      tile.number_token = 0;
      tile.robber = true;
      current_robber_tile = &tile;
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
  int token_1, token_2;
  bool first_column, last_column = false;
  for (int row = 0; row < tile_rows; row++) {
    for (int column = 0; column < tiles_in_row[row]; column++) {
      token_1 = tiles[row][column].number_token;

      if (column == 0) {
        first_column = true;
      }
      else {
        first_column = false;
      }
      if (column == (tiles_in_row[row] - 1)) {
        last_column = true;
      }
      else {
        last_column = false;
      }

      // Compare to the right
      if (!last_column) {
        if (compare_number_tokens(token_1, tiles[row][column + 1].number_token)) {
          return false;
        }
      }

      if (row != tile_rows - 1) {
        // Compare below (Assume board maximally changes one tile per row)
        if (!(tile_diff[row] < 0 && last_column)) {
          if (compare_number_tokens(token_1, tiles[row + 1][column].number_token)) {
            return false;
          }
        }

        // Compare below shifted (Assume board maximally changes one tile per row)
        if (!(tile_diff[row] < 0 && first_column)) {
          if (compare_number_tokens(token_1, tiles[row + 1][column + tile_diff[row]].number_token)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

void Board::Randomize() {
  bool correct = false;
  while (!correct) {
    ShuffleTilesAndTokens();
    AddTileTypeAndNumberTokensToTiles();
    correct = CheckNumberTokens();
  }
}

/*
 * Adds harbor types to pre-defined corners of selected tile_array.
 */
void Board::AddHarbors() {
  for (auto harbor : harbors) {
    tile_array[harbor.tile_id].corners[harbor.corners[0]]->harbor = harbor.type;
    tile_array[harbor.tile_id].corners[harbor.corners[1]]->harbor = harbor.type;
  }
}

void Board::Reset() {
  // Reset Corners
  for (auto & corner_i : corner_array) {
    corner_i.occupancy = CornerOccupancy::EmptyCorner;
    corner_i.color = Color::NoColor;
  }

  // Reset Streets
  for (auto & street_i : street_array) {
    street_i.color = Color::NoColor;
  }
}

/*
 * Prints board to the console.
 * */
void Board::PrintBoard() {
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

  int current_tile = 0;

  // Test corners
   tile_array[0].corners[1]->occupancy = Village;
   tile_array[3].corners[3]->occupancy = Village;
   tile_array[6].corners[2]->occupancy = City;
   tile_array[10].corners[4]->occupancy = City;
   tile_array[15].corners[5]->occupancy = Village;

  int current_corner_row = 0;
  int current_corner_column = 1;
  bool odd_row = true;

  for (int char_i = 0; char_i < 884; ++char_i) {
    switch (board_chars[char_i]) {

      case 'X':

        board_chars[char_i] = tile_shortnames[tile_array[current_tile].type];

        if (tile_array[current_tile].robber) {
          char_i++;
          board_chars[char_i] = '_';
          char_i++;
          board_chars[char_i] = 'R';
        } else if (tile_array[current_tile].type == Desert) {
          char_i++;
          board_chars[char_i] = '_';
          char_i++;
          board_chars[char_i] = '_';
        } else if (tile_array[current_tile].number_token < 10) {
          char_i += 2;
          board_chars[char_i] = '0' + tile_array[current_tile].number_token;
        } else {
          char_i++;
          board_chars[char_i] = '1';
          char_i++;
          board_chars[char_i] = '0' + (tile_array[current_tile].number_token - 10);
        }

        current_tile++;

        break;

      case '.':

        board_chars[char_i] = corner_shortnames[corners[current_corner_row][current_corner_column].occupancy];

        current_corner_column += 2;

        // If at the end of the row
        if (current_corner_column >= corners_in_row[current_corner_row]) {
          // Check if northern or southern hemisphere
          if (current_corner_row < 3) {
            // If we have done the even row continue to next row
            if (!odd_row) {
              if (current_corner_row == 2) {
                odd_row = false;
                current_corner_column = (int) 0;
              } else {
                odd_row = !odd_row;
                current_corner_column = (int) odd_row;
              }
              current_corner_row++;
            } else { // Otherwise restart this row but now the other numbers
              odd_row = !odd_row;
              current_corner_column = (int) odd_row;
            }
          } else { // Same but inverted for other hemisphere
            if (odd_row) {
              odd_row = !odd_row;
              current_corner_column = (int) !odd_row;
              current_corner_row++;
            } else {
              odd_row = !odd_row;
              current_corner_column = (int) !odd_row;
            }
          }
        }

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

