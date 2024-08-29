#ifndef FASTCATAN_BEAN_HELPER_H
#define FASTCATAN_BEAN_HELPER_H

#include <vector>

#include "ai_helper.h"
#include "../AIPlayer/ai_bean_player.h"

class BeanHelper : public AIHelper{
public:
  BeanHelper(unsigned int, unsigned int, unsigned int);
  ~BeanHelper();

  void update(Game* game, int id);

  void shuffle_players();
  void eliminate();
  void reproduce();
  void mutate();

  float survival_rate = 0.25;
  unsigned int survival_amount = 0;

  int mutation_length = 10000;

  std::vector<BeanNN*> nn_vector;

private:
  std::random_device rd;
  std::mt19937 gen;

  BeanNN** survived_players;

  unsigned int seed = 0;

  void bubble_sort_indices(int*, const float*, int);
};

#endif //FASTCATAN_BEAN_HELPER_H
