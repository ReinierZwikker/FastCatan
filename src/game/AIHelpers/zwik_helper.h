#ifndef FASTCATAN_ZWIK_HELPER_H
#define FASTCATAN_ZWIK_HELPER_H

#include "ai_helper.h"

class ZwikHelper : public AIHelper{
public:
  ZwikHelper(unsigned int, unsigned int);
  ~ZwikHelper();

  void update(Game*);

private:
  void eliminate();
  void reproduce();
  void mutate();

  // TODO: Add AI
};

#endif //FASTCATAN_ZWIK_HELPER_H
