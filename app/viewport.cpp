#include "viewport.h"
#include "../app/3rd_party/imgui/imgui.h"
#include "../app/3rd_party/imgui/imgui_internal.h"
#include <GL/GL.h>
#include <GL/glu.h>
#define STB_IMAGE_IMPLEMENTATION

#include "../app/3rd_party/stb_image.h" // Currently not in use
#include "iostream"
#include <mutex>

std::mutex viewport_mutex;


ViewPort::ViewPort() {

}

float ViewPort::CalculateTileShift(float shift, int row, Board* board) const {
  if (row == 0) {
    shift = 0;
  }
  else if (board->tile_diff[row - 1] < 0) {
    shift += x_spacing / 2;
  }
  else {
    shift -= x_spacing / 2;
  }
  return shift;
}

float ViewPort::ConvertTileColumn2x(int column, float shift) const {
  return (float)column * x_spacing + shift - x_spacing;
}

float ViewPort::ConvertTileRow2y(int row) const {
  return -(float)row * y_spacing + 2 * y_spacing;
}

float ViewPort::ConvertCornerColumn2x(int column, float shift) const {
  return -1.5f * x_spacing + (float) column * 0.5f * x_spacing + shift;
}

float ViewPort::ConvertCornerRow2y(int column, int row, bool increasing) const {
  if (increasing) {
    if (column % 2 == 0) {
      return 2.35f * y_spacing - (float) row * y_spacing;
    } else {
      return 2.38f * y_spacing - (float) row * y_spacing + 0.5f * y_scale;
    }
  }
  else {
    if (column % 2 == 0) {
      return 2.38f * y_spacing - (float) row * y_spacing + 0.5f * y_scale;
    } else {
      return 2.35f * y_spacing - (float) row * y_spacing;
    }
  }
}

void ViewPort::CalculateCoordinates(Game* game) {
  float x, y, tile_shift;
  float previous_x, previous_y;

  for (int row = 0; row < tile_rows; row++) {
    // Shift x location based on row to interlock the hexagons
    tile_shift = CalculateTileShift(tile_shift, row, &game->board);

    for (int column = 0; column < tiles_in_row[row]; column++) {
      game->board.tiles[row][column].coordinates[0] = ConvertTileColumn2x(column, tile_shift);
      game->board.tiles[row][column].coordinates[1] = ConvertTileRow2y(row);
    }
  }

  tile_shift = 0;
  for (int row = 0; row < corner_rows; row++) {
    tile_shift = CalculateTileShift(tile_shift, row, &game->board);
    for (int column = 0; column < corners_in_row[row]; column++) {
      if (row < 3) {
        x = ConvertCornerColumn2x(column, tile_shift);
        y = ConvertCornerRow2y(column, row, true);
      } else {
        x = ConvertCornerColumn2x(column, tile_shift - x_spacing/2);
        y = ConvertCornerRow2y(column, row, false);
      }
      game->board.corners[row][column].coordinates[0] = x;
      game->board.corners[row][column].coordinates[1] = y;
    }
  }
}

void SetTileColor(Tile* tile) {
  switch(tile->type) {
    case TileType::Hills:
      tile->color[0] = 0.835f;
      tile->color[1] = 0.231f;
      tile->color[2] = 0.078f;
      break;
    case TileType::Forest:
      tile->color[0] = 0.173f;
      tile->color[1] = 0.490f;
      tile->color[2] = 0.012f;
      break;
    case TileType::Mountains:
      tile->color[0] = 0.437f;
      tile->color[1] = 0.441f;
      tile->color[2] = 0.433f;
      break;
    case TileType::Fields:
      tile->color[0] = 0.941f;
      tile->color[1] = 0.913f;
      tile->color[2] = 0.129f;
      break;
    case TileType::Pasture:
      tile->color[0] = 0.674f;
      tile->color[1] = 0.882f;
      tile->color[2] = 0.314f;
      break;
    case TileType::Desert:
      tile->color[0] = 0.964f;
      tile->color[1] = 0.923f;
      tile->color[2] = 0.568f;
      break;
    default:
      tile->color[0] = 0.0f;
      tile->color[1] = 0.5f;
      tile->color[2] = 0.5f;
      break;
  }
}

void ViewPort::NewMap(Game* game) {
  for (auto & tile_i : game->board.tile_array) {
    SetTileColor(&tile_i);
  }
}


void ViewPort::DrawTile(float x, float y, Tile tile) const {
  // Fill
  glPushMatrix();
    glTranslatef(x, y, 0.0);
    glScalef(x_scale, y_scale, 0.0);
    glBegin(GL_POLYGON);
      glColor3f(tile.color[0], tile.color[1], tile.color[2]);
      glVertex2f( 0.0f,  tile_half_height);
      glVertex2f( tile_half_width,  0.5f * tile_half_height);
      glVertex2f( tile_half_width, -0.5f * tile_half_height);
      glVertex2f( 0.0f, -tile_half_height);
      glVertex2f(-tile_half_width, -0.5f * tile_half_height);
      glVertex2f(-tile_half_width,  0.5f * tile_half_height);
    glEnd();
  glPopMatrix();
  // Border
  glPushMatrix();
    glTranslatef(x, y, 0.0);
    glScalef(x_scale + 0.005f, y_scale + 0.005f, 0.0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(4);

    glBegin(GL_POLYGON);
      glColor3f(0.7f, 0.7f, 0.4f);
      glVertex2f( 0.0f,  tile_half_height);
      glVertex2f( tile_half_width,  0.5f * tile_half_height);
      glVertex2f( tile_half_width, -0.5f * tile_half_height);
      glVertex2f( 0.0f, -tile_half_height);
      glVertex2f(-tile_half_width, -0.5f * tile_half_height);
      glVertex2f(-tile_half_width,  0.5f * tile_half_height);
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glPopMatrix();
  // Number
  glPushMatrix();
    glTranslatef(x, y, 0.0);
    glScalef(x_scale * 0.4f, y_scale * 0.4f, 0.0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(2);

    glBegin(GL_LINE_STRIP);
    glColor3f(0.0f, 0.0f, 0.0f);
    switch (tile.number_token) {
      case 2:
        glVertex2f(-0.3f,  0.5f);
        glVertex2f( 0.3f,  0.5f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f(-0.3f,  0.0f);
        glVertex2f(-0.3f, -0.5f);
        glVertex2f( 0.3f, -0.5f);
        break;
      case 3:
        glVertex2f(-0.3f,  0.5f);
        glVertex2f( 0.3f,  0.5f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f(-0.3f,  0.0f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f( 0.3f, -0.5f);
        glVertex2f(-0.3f, -0.5f);
        break;
      case 4:
        glVertex2f(-0.3f,  0.5f);
        glVertex2f(-0.3f,  0.0f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f( 0.3f,  0.5f);
        glVertex2f( 0.3f, -0.5f);
        break;
      case 5:
        glVertex2f( 0.3f,  0.5f);
        glVertex2f(-0.3f,  0.5f);
        glVertex2f(-0.3f,  0.0f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f( 0.3f, -0.5f);
        glVertex2f(-0.3f, -0.5f);
        break;
      case 6:
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex2f( 0.3f,  0.5f);
        glVertex2f(-0.3f,  0.5f);
        glVertex2f(-0.3f,  0.0f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f( 0.3f, -0.5f);
        glVertex2f(-0.3f, -0.5f);
        glVertex2f(-0.3f,  0.0f);
        break;
      case 8:
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f( 0.3f,  0.5f);
        glVertex2f(-0.3f,  0.5f);
        glVertex2f(-0.3f,  0.0f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f( 0.3f, -0.5f);
        glVertex2f(-0.3f, -0.5f);
        glVertex2f(-0.3f,  0.0f);
        break;
      case 9:
        glVertex2f( 0.3f,  0.0f);
        glVertex2f( 0.3f,  0.5f);
        glVertex2f(-0.3f,  0.5f);
        glVertex2f(-0.3f,  0.0f);
        glVertex2f( 0.3f,  0.0f);
        glVertex2f( 0.3f, -0.5f);
        glVertex2f(-0.3f, -0.5f);
        break;
      case 10:
        glVertex2f( -0.5f,  0.5f);
        glVertex2f( -0.5f, -0.5f);
        glEnd();
        glBegin(GL_LINE_STRIP);
        glVertex2f( 0.0f,  0.5f);
        glVertex2f( 0.6f,  0.5f);
        glVertex2f( 0.6f, -0.5f);
        glVertex2f( 0.0f, -0.5f);
        glVertex2f( 0.0f,  0.5f);
        break;
      case 11:
        glVertex2f( -0.3f,  0.5f);
        glVertex2f( -0.3f, -0.5f);
        glEnd();
        glBegin(GL_LINE_STRIP);
        glVertex2f( 0.3f,  0.5f);
        glVertex2f( 0.3f, -0.5f);
        break;
      case 12:
        glVertex2f( -0.5f,  0.5f);
        glVertex2f( -0.5f, -0.5f);
        glEnd();
        glBegin(GL_LINE_STRIP);
        glVertex2f( 0.0f,  0.5f);
        glVertex2f( 0.6f,  0.5f);
        glVertex2f( 0.6f,  0.0f);
        glVertex2f( 0.0f,  0.0f);
        glVertex2f( 0.0f, -0.5f);
        glVertex2f( 0.6f, -0.5f);
        break;
    }
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glPopMatrix();
}

void ViewPort::DrawTileSelection(int id, Game* game) const {
  float x = game->board.tile_array[id].coordinates[0];
  float y = game->board.tile_array[id].coordinates[1];

  glPushMatrix();
    glTranslatef(x, y, 0.0);
    glScalef(x_scale + 0.005f, y_scale + 0.005f, 0.0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(4);

    glBegin(GL_POLYGON);
      glColor3f(0.8f, 0.0f, 0.5f);
      glVertex2f( 0.0f,  tile_half_height);
      glVertex2f( tile_half_width,  0.5f * tile_half_height);
      glVertex2f( tile_half_width, -0.5f * tile_half_height);
      glVertex2f( 0.0f, -tile_half_height);
      glVertex2f(-tile_half_width, -0.5f * tile_half_height);
      glVertex2f(-tile_half_width,  0.5f * tile_half_height);
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glPopMatrix();
}

void ViewPort::DrawCorner(float x, float y, Corner corner) const {
  viewport_mutex.lock();
  CornerOccupancy corner_occupancy = corner.occupancy;
  viewport_mutex.unlock();

  if (corner_occupancy != CornerOccupancy::EmptyCorner) {
    glPushMatrix();
    glTranslatef(x, y, 0.0);
    glScalef(x_scale * 0.3f, y_scale * 0.3f, 0.0);
    glBegin(GL_POLYGON);

    switch (corner.color) {
      case Color::Blue:
        glColor3f(0.0f, 0.0f, 1.0f);
        break;
      case Color::Green:
        glColor3f(0.0f, 1.0f, 0.0f);
        break;
      case Color::Red:
        glColor3f(1.0f, 0.0f, 0.0f);
        break;
      case Color::White:
        glColor3f(0.9f, 0.9f, 0.9f);
        break;
      case Color::NoColor:
        glColor3f(0.0f, 0.0f, 0.0f);
        break;
    }

    switch (corner_occupancy) {
      case CornerOccupancy::Village:
        glVertex2f(-0.5f, -0.7f);
        glVertex2f(-0.5f,  0.2f);
        glVertex2f( 0.0f,  0.9f);
        glVertex2f( 0.5f,  0.2f);
        glVertex2f( 0.5f, -0.7f);
        glEnd();
        glPopMatrix();
        break;
      case CornerOccupancy::City:
        glVertex2f(-0.6f, -1.0f);
        glVertex2f(-0.6f,  0.6f);
        glVertex2f(-0.3f,  1.0f);
        glVertex2f( 0.0f,  0.6f);
        glVertex2f( 0.0f,  0.0f);
        glVertex2f( 0.6f,  0.0f);
        glVertex2f( 0.6f, -1.0f);
        glEnd();
        glPopMatrix();
        break;
    }
  }
}

void ViewPort::DrawCornerSelection(int id, Game* game) const {
  float x = game->board.corner_array[id].coordinates[0];
  float y = game->board.corner_array[id].coordinates[1];

  glPushMatrix();
    glTranslatef(x, y - 0.01f, 0.0);
    glScalef(x_scale - 0.01f, y_scale - 0.01f, 0.0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(4);

    glBegin(GL_POLYGON);
      glColor3f(0.8f, 0.0f, 0.5f);
      for (int i = 0; i < 20; i++) {
        glVertex2f(0.3f * cos((float) i * (3.14f/10)), 0.5f * sin((float) i * (3.14f/10)));
      }
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glPopMatrix();
}

void ViewPort::DrawCornerPlayerSelection(int id, Game* game, CornerOccupancy occupancy) const {
  float x = game->board.corner_array[id].coordinates[0];
  float y = game->board.corner_array[id].coordinates[1];

  glPushMatrix();
  glTranslatef(x, y, 0.0);
  glScalef(x_scale * 0.3f, y_scale * 0.3f, 0.0);
  glBegin(GL_POLYGON);

  glColor3f(0.8f, 0.0f, 0.5f);

  switch (occupancy) {
    case CornerOccupancy::Village:
      glVertex2f(-0.5f, -0.7f);
      glVertex2f(-0.5f,  0.2f);
      glVertex2f( 0.0f,  0.9f);
      glVertex2f( 0.5f,  0.2f);
      glVertex2f( 0.5f, -0.7f);
      glEnd();
      glPopMatrix();
      break;
    case CornerOccupancy::City:
      glVertex2f(-0.6f, -1.0f);
      glVertex2f(-0.6f,  0.6f);
      glVertex2f(-0.3f,  1.0f);
      glVertex2f( 0.0f,  0.6f);
      glVertex2f( 0.0f,  0.0f);
      glVertex2f( 0.6f,  0.0f);
      glVertex2f( 0.6f, -1.0f);
      glEnd();
      glPopMatrix();
      break;
    default:
      glEnd();
      glPopMatrix();
  }
}

void ViewPort::DrawStreet(int id, Game* game) const {
  viewport_mutex.lock();
  Color street_color = game->board.street_array[id].color;
  viewport_mutex.unlock();

  if (street_color != Color::NoColor) {
    glPushMatrix();
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glLineWidth(3);

      glBegin(GL_LINES);

        switch (street_color) {
          case Color::Blue:
            glColor3f(0.0f, 0.0f, 1.0f);
            break;
          case Color::Green:
            glColor3f(0.0f, 1.0f, 0.0f);
            break;
          case Color::Red:
            glColor3f(1.0f, 0.0f, 0.0f);
            break;
          case Color::White:
            glColor3f(0.9f, 0.9f, 0.9f);
            break;
        }

        float x_1 = game->board.street_array[id].corners[0]->coordinates[0];
        float y_1 = game->board.street_array[id].corners[0]->coordinates[1];
        float x_2 = game->board.street_array[id].corners[1]->coordinates[0];
        float y_2 = game->board.street_array[id].corners[1]->coordinates[1];

        glVertex2f(x_1, y_1 - lower_road);
        glVertex2f(x_2, y_2 - lower_road);

      glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glPopMatrix();
  }
}

void ViewPort::DrawStreetSelection(int id, Game* game) const {
  glPushMatrix();
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glLineWidth(3);

  glBegin(GL_LINES);

  glColor3f(0.8f, 0.0f, 0.5f);

  float x_1 = game->board.street_array[id].corners[0]->coordinates[0];
  float y_1 = game->board.street_array[id].corners[0]->coordinates[1];
  float x_2 = game->board.street_array[id].corners[1]->coordinates[0];
  float y_2 = game->board.street_array[id].corners[1]->coordinates[1];

  glVertex2f(x_1, y_1 - lower_road);
  glVertex2f(x_2, y_2 - lower_road);

  glEnd();

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glPopMatrix();
}

void ViewPort::Refresh(Game* game) {
  float x, y;

  // Render the Tiles
  for (auto & tile_i : game->board.tile_array) {
    x = tile_i.coordinates[0];
    y = tile_i.coordinates[1];
    DrawTile(x, y, tile_i);
  }

  // Render the Tile Selection
  if (tile_selection_item.render) {
    DrawTileSelection(tile_selection_item.id, tile_selection_item.game);
    tile_selection_item.render = false;
  }

  // Render the Streets
  for (auto & street : game->board.street_array) {
    DrawStreet(street.id, game);
  }

  // Render the Street Selection
  if (street_selection_item.render) {
    DrawStreetSelection(street_selection_item.id, street_selection_item.game);
    street_selection_item.render = false;
  }
  if (player_street_selection_item.render) {
    DrawStreetSelection(player_street_selection_item.id, player_street_selection_item.game);
    player_street_selection_item.render = false;
  }

  // Render the Corners
  for (auto & corner_i : game->board.corner_array) {
    x = corner_i.coordinates[0];
    y = corner_i.coordinates[1];
    DrawCorner(x, y, corner_i);
  }

  // Render the Corner Selection
  if (corner_selection_item.render) {
    DrawCornerSelection(corner_selection_item.id, corner_selection_item.game);
    corner_selection_item.render = false;
  }
  if (player_corner_selection_item.render) {
    DrawCornerPlayerSelection(player_corner_selection_item.id, player_corner_selection_item.game, player_corner_selection_item.corner_occupancy);
    player_corner_selection_item.render = false;
  }

}

