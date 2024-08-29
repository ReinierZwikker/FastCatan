#ifndef FASTCATAN_ZWIK_HELPER_H
#define FASTCATAN_ZWIK_HELPER_H

#include "ai_helper.h"
#include "../AIPlayer/ai_zwik_player.h"
#include "../player.h"

#include <atomic>
#include <vector>
#include <string>

class ZwikHelper : public AIHelper{
public:
  ZwikHelper(unsigned int pop_size, unsigned int num_threads);
  ~ZwikHelper();

  void update(Game*);
  Player* get_new_player(Board* board, Color player_color);
  void update_epoch();


private:

  std::mutex helper_mutex;

  std::mt19937 gen;

  int population_remain_size, population_unfit_size;

  struct Individual {
      explicit Individual(int seed, int given_id) {
        gene = NeuralWeb(AIZwikPlayer::amount_of_neurons,
                         AIZwikPlayer::amount_of_env_inputs + AIZwikPlayer::amount_of_inputs,
                         AIZwikPlayer::amount_of_outputs,
                         seed).to_string();
        score = 0.0f;
        age = 0;
        id = given_id;
      }
      Individual(const std::string& parent_A_gene,
                 const std::string& parent_B_gene,
                 int seed,
                 int given_id) {
        gene = NeuralWeb(parent_A_gene, parent_B_gene, seed).to_string();
        score = 0.0f;
        age = 0;
        id = given_id;
      }

      std::string gene;
      float score;
      int age;
      int id;
  };

  std::vector<Individual> gene_pool{};

  std::vector<Player*> current_players{};

  inline static bool score_comp(const Individual& A, const Individual& B)
    { return A.score > B.score; };


public:
  inline void sort_genes() { std::stable_sort(gene_pool.begin(), gene_pool.end(), score_comp); };

  inline Individual get_gene(const int id) { return gene_pool[id]; }

  void store_gene(int gene_i,
                  const std::string& filename,
                  const std::filesystem::path& dirPath);
};

#endif //FASTCATAN_ZWIK_HELPER_H
