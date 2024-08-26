#ifndef FASTCATAN_NEURALWEB_H
#define FASTCATAN_NEURALWEB_H

#define NEURON_QUEUE_SIZE 5000
#define NEURON_DOWNSTREAM_SIZE 5

#include <random>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>

struct Neuron {
  Neuron(int set_id, float set_threshold);

  int id;

  float value, threshold;

  void update(float in_value);

  Neuron *downstream_connections[NEURON_DOWNSTREAM_SIZE] {};
  float downstream_weights[NEURON_DOWNSTREAM_SIZE] {};

  ~Neuron();
};


struct QueuedNeuron {
    inline QueuedNeuron(Neuron *neuron_to_queue, float value_to_send)
             { neuron = neuron_to_queue; sent_value = value_to_send; }

    Neuron *neuron;
    float sent_value;
};

/*
 * Replaced with standard queue
class NeuronQueuePtr {

    QueuedNeuron *queue;

//public:

    int current_head = 0;
    int current_tail = 0;

public:
    NeuronQueuePtr();
    ~NeuronQueuePtr();

    int run_next_neuron();

    void add_to_queue(Neuron *neuron, float value);
};
*/

class NeuralWeb {

public:
  NeuralWeb(int amount_of_neurons,
            int amount_of_inputs,
            int amount_of_outputs,
            unsigned int web_seed,
            float random_threshold_min = 0.8f,
            float random_threshold_max = 0.99f,
            float random_weight_min = 0.1f,
            float random_weight_max = 0.7f,
            int random_am_of_conn_min = 1);
  NeuralWeb(const std::string& ai_str);
  NeuralWeb(const std::string& ai_str_A,
            const std::string& ai_str_B,
            const int seed);
  NeuralWeb(const std::string& filename,
            const std::filesystem::path& dirPath);
  ~NeuralWeb();


  int run_web(float *inputs, float *outputs, int max_cycles);

  void to_json(const std::string& filename,
               const std::filesystem::path& dirPath);

  std::string to_string();

  void to_string(const std::string& filename,
                 const std::filesystem::path& dirPath);

  void combine_strings(const std::string& ai_str_A, const std::string& ai_str_B);

private:

  void from_string(const std::string& ai_str);

  Neuron *neurons;

  Neuron  **input_neurons;
  Neuron **output_neurons;

  // NeuronQueuePtr neuron_queue;
  std::queue<QueuedNeuron> queue;


  std::random_device randomDevice;
  std::mt19937 gen;

  int AmountOfNeurons,
      AmountOfInputs,
      AmountOfOutputs;



};


#endif //FASTCATAN_NEURALWEB_H
