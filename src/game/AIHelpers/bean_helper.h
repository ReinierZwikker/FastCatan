#ifndef FASTCATAN_BEAN_HELPER_H
#define FASTCATAN_BEAN_HELPER_H

#include <vector>

#include "ai_helper.h"
#include "../AIPlayer/ai_bean_player.h"

class BeanHelper : public AIHelper{
public:
  BeanHelper(unsigned int, unsigned int, unsigned int);
  ~BeanHelper();

  void update(Game* game, int id, int game_i);

  void shuffle_players();
  void eliminate();
  void reproduce();
  void mutate();

  float average_points_mult = 1;
  float win_rate_mult = 100;
  float average_moves_mult = 0.005;

  int mutation_length = 10000;

private:
  std::random_device rd;
  std::mt19937 gen;

  std::vector<BeanNN*> bean_nn_vector;

  BeanNN** survived_players;

  unsigned int seed = 0;

  void bubble_sort_indices(int*, const float*, int);
};

#endif //FASTCATAN_BEAN_HELPER_H
