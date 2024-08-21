#ifndef FASTCATAN_BEAN_HELPER_H
#define FASTCATAN_BEAN_HELPER_H

#include <vector>

#include "ai_helper.h"
#include "../AIPlayer/ai_bean_player.h"

class BeanHelper : public AIHelper{
public:
  BeanHelper(unsigned int, uint8_t, uint8_t, unsigned int, unsigned int);
  ~BeanHelper();

  void update(Game* game, int id);

private:
  void eliminate();
  void reproduce();
  void mutate();

  std::random_device rd;
  std::mt19937 gen;

  std::vector<BeanNN*> bean_nn_vector;

  uint8_t amount_of_layers = 0;
  uint8_t nodes_in_layer = 0;
  unsigned int seed = 0;
};

#endif //FASTCATAN_BEAN_HELPER_H
