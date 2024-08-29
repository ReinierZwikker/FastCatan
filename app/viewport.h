#ifndef FASTCATAN_VIEWPORT_H
#define FASTCATAN_VIEWPORT_H

#include <windows.h>
#include <GL/GL.h>
#include <GL/glu.h>

#include "../src/game/game.h"

// Used to delay the selection render
struct SelectionItem {
  int id;
  Game* game;
  CornerOccupancy corner_occupancy;
  bool render;
};


class ViewPort {
  public:
    ViewPort();

    void NewMap(Game*);
    void CalculateCoordinates(Game* game);
    void Refresh(Game*);

    SelectionItem tile_selection_item{};
    SelectionItem corner_selection_item{};
    SelectionItem player_corner_selection_item{};
    SelectionItem street_selection_item{};
    SelectionItem player_street_selection_item{};

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
    void DrawTileSelection(int, Game*) const;
    void DrawCorner(float, float, Corner) const;
    void DrawCornerSelection(int, Game*) const;
    void DrawCornerPlayerSelection(int id, Game *game, CornerOccupancy occupancy) const;
    void DrawStreet(int, Game*) const;
    void DrawStreetSelection(int id, Game* game) const;

};

#endif //FASTCATAN_VIEWPORT_H
