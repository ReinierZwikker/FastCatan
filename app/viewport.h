#ifndef FASTCATAN_VIEWPORT_H
#define FASTCATAN_VIEWPORT_H

#include <GL/GL.h>
#include <GL/glu.h>

#include "../src/game/game.h"

// Used to delay the tile selection render
struct TileSelectionItem {
  int id;
  Game* game;
  bool render;
};

// Use to delay the corner selection render
struct CornerSelectionItem {
  int id;
  Game* game;
  bool render;
};

class ViewPort {
  public:
    ViewPort();

    static void NewMap(Game*);
    void Refresh(Game*);

    TileSelectionItem tile_selection_item{};
    CornerSelectionItem corner_selection_item{};

  private:
    float x_spacing = 0.13f;
    float y_spacing = 0.16f;

    float tile_half_width = 0.6f;
    float tile_half_height = 1.0f;

    float x_scale = 0.1f;
    float y_scale = 0.1f;

    float lower_road = 0.005f;

    float CalculateTileShift(float shift, int row, Board *board) const;
    float ConvertTileColumn2x(int, float) const;
    float ConvertTileRow2y(int) const;
    float ConvertCornerColumn2x(int, float) const;
    float ConvertCornerRow2y(int, int, bool) const;

    void DrawTile(float, float, Tile) const;
    void DrawTileSelection(int, Game*);
    void DrawCorner(float, float, Corner) const;
    void DrawCornerSelection(int, Game*);
    void DrawStreet(float, float, float, float, Street) const;

};

#endif //FASTCATAN_VIEWPORT_H
