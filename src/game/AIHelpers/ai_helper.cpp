#include "ai_helper.h"


AIHelper::AIHelper(unsigned int pop_size) {
  if (pop_size > pow(2, sizeof(AISummary::id) * 8)) {
    throw std::invalid_argument("Population size is too big");
  }
  population_size = pop_size;
}

AIHelper::~AIHelper() {

}
