#ifndef FASTCATAN_AI_HELPER_H
#define FASTCATAN_AI_HELPER_H

#include <iostream>

#include "../player.h"
#include "../components.h"

class AIHelper {
public:
  AIHelper(unsigned int);
  ~AIHelper();

  unsigned int population_size = 0;
  Player* ai_players[4];
  AISummary* ai_summaries[4]{};
};

#endif //FASTCATAN_AI_HELPER_H
