#include "zwik_helper.h"

#include <ctime>
#include <algorithm>
#include <vector>

ZwikHelper::ZwikHelper(unsigned int pop_size, unsigned int num_threads) : AIHelper(pop_size, num_threads), gen((int) time(nullptr)) {
  if (pop_size > pow(2, sizeof(AISummary::id) * 8)) {
    throw std::invalid_argument("Population size is too big");
  }
  population_size = pop_size;
  population_remain_size = (int) (0.3f * (float) population_size);
  population_unfit_size = (int) (0.3f * (float) population_size);
  number_of_threads = num_threads;

  gene_pool.reserve(population_size);
  std::uniform_int_distribution<> random_seed(0, 100 * (int) population_size);

  // Start with random gene pool
  for (int pop_i = 0; pop_i < population_size; ++pop_i) {
    gene_pool.emplace_back(random_seed(gen), pop_i);
  }

  current_players.reserve(number_of_threads);
}

ZwikHelper::~ZwikHelper() {
  for (auto & current_player : current_players) {
    delete current_player;
  }
  current_players.clear();
  gene_pool.clear();
}

void ZwikHelper::update(Game* game) {
  for (auto player : game->players) {
    if (player->agent->get_player_type() == PlayerType::zwikPlayer) {
      float obtained_score = 0.0f;

      // Add score for each victory point
      obtained_score += 0.5f * (float) player->victory_points;

      // Add score for winning player
      if (game->game_winner == player->player_color) {
        obtained_score += 10.0f;
      }

      // Add to score if under 20 rounds, subtract if above
      if (game->current_round < 20) {
        obtained_score += (30.0f - (float) game->current_round) / 3.0f;
      } else {
        obtained_score += (30.0f - (float) game->current_round) / 15.0f;
      }

      // Subtract score for mistakes in choosing a move
      obtained_score -= 0.5f * (float) *(int *) player->agent->get_custom_player_attribute(1);

      helper_mutex.lock();
      gene_pool[player->player_id].score += obtained_score;
      gene_pool[player->player_id].played_games++;
      helper_mutex.unlock();
    }
  }
}

Player *ZwikHelper::get_new_player(Board* board, Color player_color) {
  std::uniform_int_distribution<> random_player(0, (int) population_size - 1);
  int selected_ai = random_player(gen);

  helper_mutex.lock();
  current_players.push_back(new Player(board, player_color, selected_ai));
  current_players.back()->agent = new AIZwikPlayer(current_players.back(), gene_pool[selected_ai].gene);
  current_players.back()->activated = true;

  Player* latest_player = current_players.back();
  helper_mutex.unlock();

  return latest_player;
}

void add_float_to_csv(float value,
                      const std::string& filename,
                      const std::filesystem::path& dirPath) {
  if (!std::filesystem::exists(dirPath)) {
    std::filesystem::create_directory(dirPath);
  }

  std::ofstream file;
  file.open(dirPath.string() + "/" + filename, std::ios_base::app);
  file << std::to_string(value) + ", ";
  file.close();
}

void ZwikHelper::update_epoch() {

  // Add average score to csv-file
  float sum_score = 0.0f;
  for (auto &individual : gene_pool) {
    sum_score += individual.score;
  }
  add_float_to_csv(sum_score / ((float) population_size),
                   "training_1.csv",
                   (std::filesystem::path) "trainings");

  // Correct for amount of games played
  for (auto &individual : gene_pool) {
    individual.score /= (float) individual.played_games;
  }

  // find player ranking
  sort_genes();

  // Leave top performers
  for (int gene_i = 0; gene_i < population_remain_size; ++gene_i) {
    gene_pool[gene_i].id = gene_i;
    ++gene_pool[gene_i].age;
    gene_pool[gene_i].score = 0.0f;
    gene_pool[gene_i].played_games = 0;
  }

  std::vector<Individual> new_individuals;

  std::uniform_int_distribution<> random_parent(0, (int) population_size - population_unfit_size);
  for (int gene_i = population_remain_size; gene_i < gene_pool.size(); ++gene_i) {
    new_individuals.emplace_back(gene_pool[random_parent(gen)].gene,
                                 gene_pool[random_parent(gen)].gene,
                                 random_parent(gen),
                                 gene_i);
  }
  for (int gene_i = population_remain_size; gene_i < gene_pool.size(); ++gene_i) {
    gene_pool[gene_i] = new_individuals[gene_i - population_remain_size];
  }

  for (int player_i; player_i < current_players.size(); ++player_i) {
    delete current_players[player_i];
  }
  current_players.clear();
}

void ZwikHelper::store_gene(int gene_i,
                            const std::string& filename,
                            const std::filesystem::path& dirPath) {
  if (!std::filesystem::exists(dirPath)) {
    std::filesystem::create_directory(dirPath);
  }
 std::ofstream file(dirPath.string() + "/" + filename);
  file << gene_pool[gene_i].gene;
  file.close();
}
