//
// Created by reini on 25/04/2024.
//

#ifndef FASTCATAN_BOARD_H
#define FASTCATAN_BOARD_H

enum tile_type {
    Desert,
    Hills,
    Forest,
    Mountains,
    Fields,
    Pasture
};

enum corner_occupancy {
    EmptyCorner,
    GreenVillage,
    GreenCity,
    RedVillage,
    RedCity,
    WhiteVillage,
    WhiteCity,
    BlueVillage,
    BlueCity
};

enum street_occupancy {
    EmptyStreet,
    GreenStreet,
    RedStreet,
    WhiteStreet,
    BlueStreet,
};

struct Board {

    Board();

    tile_type tiles[18];
    int tile_number[18];
    corner_occupancy corners[53];
    street_occupancy streets[71];

    // Max amount of tiles included in the game
    const int max_terrain_tiles[6] = {1, 3, 4, 3, 4, 4};
    const tile_type tile_order[6] = {
            Desert, Hills, Forest, Mountains, Fields, Pasture
    };
};
#endif //FASTCATAN_BOARD_H
