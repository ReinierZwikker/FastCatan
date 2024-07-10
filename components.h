#ifndef FASTCATAN_COMPONENTS_H
#define FASTCATAN_COMPONENTS_H

enum colors {
    Color_None,
    Green,
    Red,
    White,
    Blue
};

enum cards {
    Brick,
    Lumber,
    Ore,
    Grain,
    Wool
};

enum tile_type {
    Desert,
    Hills,
    Forest,
    Mountains,
    Fields,
    Pasture
};

static const char* tile_names[] = {
    "Desert",
    "Hills",
    "Forest",
    "Mountains",
    "Fields",
    "Pasture"
};

static const char tile_shortnames[] = {
    'D',  // Desert
    'H',  // Hills
    'F',  // Forest
    'M',  // Mountains
    'f',  // Fields
    'P'   // Pasture
};

enum corner_occupancy {
    EmptyCorner,
    Village,
    City
};

static const char corner_shortnames[] = {
    '.',  // EmptyCorner
    'g',  // GreenVillage
    'G',  // GreenCity
    'r',  // RedVillage
    'R',  // RedCity
    'w',  // WhiteVillage
    'W',  // WhiteCity
    'b',  // BlueVillage
    'B'   // BlueCity
};

enum harbor_types {
    Harbor_None,
    Harbor_Generic,
    Harbor_Brick,
    Harbor_Grain,
    Harbor_Wool,
    Harbor_Lumber,
    Harbor_Ore
};

#endif //FASTCATAN_COMPONENTS_H
