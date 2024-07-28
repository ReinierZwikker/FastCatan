#include "viewport.h"
#include "../app/3rd_party/imgui/imgui.h"
#include "../app/3rd_party/imgui/imgui_internal.h"
#include <GL/GL.h>
#include <GL/glu.h>
#define STB_IMAGE_IMPLEMENTATION

#include "../app/3rd_party/stb_image.h" // Currently not in use
#include "iostream"

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

void ViewPort::DrawTileSelection(int id, Game* game) {
  float x, y, shift;
  int row = 0;
  int column = 0;

  for (int i = 0; i < id; i++) {
    column += 1;
    if (column == tiles_in_row[row]) {
      row += 1;
      column = 0;
      shift = CalculateTileShift(shift, row, &game->board);
    }
  }

  x = ConvertTileColumn2x(column, shift);
  y = ConvertTileRow2y(row);

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
  if (corner.occupancy != CornerOccupancy::EmptyCorner) {
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

    switch (corner.occupancy) {
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

void ViewPort::DrawCornerSelection(int id, Game* game) {
  float x, y, shift;
  int row = 0;
  int column = 0;

  for (int i = 0; i < id; i++) {
    column += 1;
    if (column == corners_in_row[row]) {
      row += 1;
      column = 0;
      shift = CalculateTileShift(shift, row, &game->board);
    }
  }

  if (row < 3) {
    x = ConvertCornerColumn2x(column, shift);
    y = ConvertCornerRow2y(column, row, true);
  } else {
    x = ConvertCornerColumn2x(column, shift - x_spacing/2);
    y = ConvertCornerRow2y(column, row, false);
  }

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

void ViewPort::DrawStreet(float x_1, float x_2, float y_1, float y_2, Street street) const {
  if (street.color != Color::NoColor) {
    glPushMatrix();
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glLineWidth(3);

      glBegin(GL_LINES);

        switch (street.color) {
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

        glVertex2f(x_1, y_1 - lower_road);
        glVertex2f(x_2, y_2 - lower_road);

      glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glPopMatrix();
  }
}

void ViewPort::Refresh(Game* game) {
  float x, y, tile_shift;
  float previous_x, previous_y;
  for (int row = 0; row < tile_rows; row++) {
    // Shift x location based on row to interlock the hexagons
    tile_shift = CalculateTileShift(tile_shift, row, &game->board);

    for (int column = 0; column < tiles_in_row[row]; column++) {
      x = ConvertTileColumn2x(column, tile_shift);
      y = ConvertTileRow2y(row);
      DrawTile(x, y, game->board.tiles[row][column]);
    }
  }

  // Render the Tile Selection
  if (tile_selection_item.render) {
    DrawTileSelection(tile_selection_item.id, tile_selection_item.game);
    tile_selection_item.render = false;
  }

  tile_shift = 0;
  float y_below;
  for (int row = 0; row < corner_rows; row++) {
    tile_shift = CalculateTileShift(tile_shift, row, &game->board);
    for (int column = 0; column < corners_in_row[row]; column++) {
      if (row < 3) {
        x = ConvertCornerColumn2x(column, tile_shift);
        y = ConvertCornerRow2y(column, row, true);
        if (column % 2 == 0) {
          y_below = ConvertCornerRow2y(column + 1, row + 1, true);
          DrawStreet(x, x, y_below, y, game->board.streets[2 * row + 1][column/2]);
        }
      } else {
        x = ConvertCornerColumn2x(column, tile_shift - x_spacing/2);
        y = ConvertCornerRow2y(column, row, false);
        if (column % 2 == 1 && row < corner_rows - 1) {
          y_below = ConvertCornerRow2y(column, row + 1, true);
          DrawStreet(x, x, y_below, y, game->board.streets[2 * row + 1][column/2]);
        }
      }

      if (column != 0) {
        DrawStreet(previous_x, x, previous_y, y, game->board.streets[2 * row][column - 1]);
      }
      previous_x = x;
      previous_y = y;
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
      DrawCorner(x, y, game->board.corners[row][column]);
      previous_x = x;
      previous_y = y;
    }
  }

  // Render the Corner Selection
  if (corner_selection_item.render) {
    DrawCornerSelection(corner_selection_item.id, corner_selection_item.game);
    corner_selection_item.render = false;
  }

}

