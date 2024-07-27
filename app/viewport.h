#ifndef FASTCATAN_VIEWPORT_H
#define FASTCATAN_VIEWPORT_H

#include <GL/GL.h>
#include <GL/glu.h>

#include "../src/game/game.h"

// Used to delay the selection render
struct TileSelectionItem {
  int id;
  Game* game;
  bool render;
};

class ViewPort {
  public:
    ViewPort();

    static void NewMap(Game*);
    void Refresh(Game*);

    void DrawTileSelection(int, Game*);

    TileSelectionItem tile_selection_item{};

  private:
    float x_spacing = 0.13f;
    float y_spacing = 0.17f;

    float tile_half_width = 0.6f;
    float tile_half_height = 1.0f;

    float sx = 0.1f;
    float sy = 0.1f;

    void DrawTile(float, float, Tile) const;

    float CalculateShift(float, int, Board*) const;
    float ConvertColumn2x(int, float) const;
    float ConvertRow2y(int) const;
};

#endif //FASTCATAN_VIEWPORT_H
