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

float ViewPort::CalculateShift(float shift, int row, Board* board) const {
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

float ViewPort::ConvertColumn2x(int column, float shift) const {
  return (float)column * x_spacing + shift - x_spacing;
}

float ViewPort::ConvertRow2y(int row) const {
  return -(float)row * y_spacing + 2 * y_spacing;
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
  glPushMatrix();
    glTranslatef(x, y, 0.0);
    glScalef(sx, sy, 0.0);
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
      shift = CalculateShift(shift, row, &game->board);
    }
  }

  x = ConvertColumn2x(column, shift);
  y = ConvertRow2y(row);

  glPushMatrix();
    glTranslatef(x, y, 0.0);
    glScalef(sx + 0.005f, sy + 0.005f, 0.0);

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


void ViewPort::Refresh(Game* game) {
  float x, y, shift;
  for (int row = 0; row < tile_rows; row++) {
    // Shift x location based on row to interlock the hexagons
    shift = CalculateShift(shift, row, &game->board);

    for (int column = 0; column < tiles_in_row[row]; column++) {
      x = ConvertColumn2x(column, shift);
      y = ConvertRow2y(row);
      DrawTile(x, y, game->board.tiles[row][column]);
    }
  }

  // Render the Tile Selection
  if (tile_selection_item.render) {
    DrawTileSelection(tile_selection_item.id, tile_selection_item.game);
    tile_selection_item.render = false;
  }

}

