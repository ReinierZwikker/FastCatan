#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include <utility>
#include "NeuralWeb.h"

/******************
 *     NEURON     *
 ******************/

Neuron::Neuron(int set_id, float set_threshold) {
  id = set_id;
  value = 0;
  threshold = set_threshold;


  for (int neuron_i = 0; neuron_i < NEURON_DOWNSTREAM_SIZE; ++neuron_i) {
    downstream_connections[neuron_i] = nullptr;
    downstream_weights[neuron_i] = 0;
  }
}

void Neuron::update(float in_value) {
  value += in_value;
}

Neuron::~Neuron() = default;


/******************
 *     QUEUE      *
 ******************/

/***
 * Pointer Version UNUSED
 */

/*
NeuronQueuePtr::NeuronQueuePtr() {
  queue = (QueuedNeuron*) malloc(NEURON_QUEUE_SIZE * sizeof(QueuedNeuron));

  for (int neuron_i = 0; neuron_i < NEURON_QUEUE_SIZE; ++neuron_i) {
    queue[neuron_i] = QueuedNeuron(nullptr, 0);
  }
}

NeuronQueuePtr::~NeuronQueuePtr() {
  free(queue);
}

int NeuronQueuePtr::run_next_neuron() {
  Neuron *current_neuron = queue[current_tail].neuron;

  if (current_neuron == nullptr) {
    return -1;
  }

  current_neuron->update(queue[current_tail].sent_value);

  if (current_neuron->value > current_neuron->threshold) {
    current_neuron->value -= current_neuron->threshold;
    for (int neuron_i = 0; neuron_i < NEURON_DOWNSTREAM_SIZE; ++neuron_i) {
      if (current_neuron->downstream_connections[neuron_i] != nullptr) {
        add_to_queue(current_neuron->downstream_connections[neuron_i],
                     current_neuron->value * current_neuron->downstream_weights[neuron_i]);
      }
    }
  }

  ++current_tail;
  if (current_tail >= NEURON_QUEUE_SIZE) { current_tail = 0; }

  return 0;
}

void NeuronQueuePtr::add_to_queue(Neuron *neuron, float value) {
  if (current_head >= NEURON_QUEUE_SIZE) { current_head = 0; }
  if (current_head == current_tail) { printf("The snake ate its tail!\n"); }

  queue[current_head] = QueuedNeuron(neuron, value);
  ++current_head;
}
*/

/******************
 *      WEB       *
 ******************/

NeuralWeb::NeuralWeb(int amount_of_neurons,
                     int amount_of_inputs,
                     int amount_of_outputs,
                     unsigned int web_seed,
                     float random_threshold_min,
                     float random_threshold_max,
                     float random_weight_min,
                     float random_weight_max,
                     int random_am_of_conn_min) : gen(web_seed) {
  AmountOfNeurons = amount_of_neurons;
  AmountOfInputs = amount_of_inputs;
  AmountOfOutputs = amount_of_outputs;

  if (AmountOfNeurons < AmountOfInputs + AmountOfOutputs) {
    throw std::invalid_argument("Not enough neurons for the amount of inputs and outputs");
  }

  neurons = (Neuron*) malloc(AmountOfNeurons * sizeof(Neuron));

  // Generate Neurons
  std::uniform_real_distribution<float> random_threshold(random_threshold_min, random_threshold_max);

  for (int neuron_i = 0; neuron_i < AmountOfNeurons; ++neuron_i) {
    neurons[neuron_i] = Neuron(neuron_i, random_threshold(gen));
  }

  // Assign random links
  std::uniform_int_distribution<int> random_neuron(0, AmountOfNeurons-1);
  std::uniform_int_distribution<int> random_amount_of_connections(random_am_of_conn_min, NEURON_DOWNSTREAM_SIZE-1);
  std::uniform_real_distribution<float> random_weight(random_weight_min, random_weight_max);

  for (int neuron_i = 0; neuron_i < AmountOfNeurons; ++neuron_i) {
    int amount_of_connections = random_amount_of_connections(gen);
    int chosen_neurons[10] = {-1, -1, -1, -1, -1,  -1,  -1,  -1,  -1, -1};
    for (int connection_i = 0; connection_i < amount_of_connections; ++connection_i) {
      bool neuron_found = false;
      int chosen_neuron;
      while (!neuron_found) {
        chosen_neuron = random_neuron(gen);
        neuron_found = true;
        for (auto prev_chosen_neuron : chosen_neurons) {
          if (prev_chosen_neuron == chosen_neuron) {
            neuron_found = false;
          }
        }
      }
      chosen_neurons[connection_i] = chosen_neuron;
      neurons[neuron_i].downstream_connections[connection_i] = &neurons[chosen_neuron];
      neurons[neuron_i].downstream_weights[connection_i] = random_weight(gen);
    }
  }

  // Link input and output neurons
  input_neurons = (Neuron**) malloc(AmountOfInputs * sizeof(Neuron*));
  // input_neurons = new Neuron*[AmountOfInputs];
  for (int input_neuron_i = 0; input_neuron_i < AmountOfInputs; ++input_neuron_i) {
    input_neurons[input_neuron_i] = &neurons[input_neuron_i];
  }

  output_neurons = (Neuron**) malloc(AmountOfOutputs * sizeof(Neuron*));
  for (int output_neuron_i = 0; output_neuron_i < AmountOfOutputs; ++output_neuron_i) {
    output_neurons[output_neuron_i] = &neurons[AmountOfInputs + output_neuron_i];
  }

}


std::string read_until(const std::string& ai_str, int &current_char, char end_marker) {

  std::string temp;

  while (ai_str[current_char] != end_marker) {
    temp += ai_str[current_char++];
  }
  current_char++;

  return temp;
}

/**
 * from string file,
 * takes filename and path to file
 */
NeuralWeb::NeuralWeb(const std::string &filename, const std::filesystem::path &dirPath) : gen(0) {
  AmountOfNeurons = 0;
  AmountOfInputs = 0;
  AmountOfOutputs = 0;
  neurons = (Neuron*) nullptr;
  input_neurons = (Neuron**) nullptr;
  output_neurons = (Neuron**) nullptr;

  std::ifstream file(dirPath.string() + "/" + filename);
  std::string file_str;
  file >> file_str;
  file.close();

  from_string(file_str);
}

/**
 * from string,
 * takes string
 */
NeuralWeb::NeuralWeb(const std::string& ai_str) : gen(0) {
  AmountOfNeurons = 0;
  AmountOfInputs = 0;
  AmountOfOutputs = 0;
  neurons = (Neuron*) nullptr;
  input_neurons = (Neuron**) nullptr;
  output_neurons = (Neuron**) nullptr;

  from_string(ai_str);
}

NeuralWeb::NeuralWeb(const std::string& ai_str_A, const std::string& ai_str_B, const int seed) : gen(seed) {
  AmountOfNeurons = 0;
  AmountOfInputs = 0;
  AmountOfOutputs = 0;
  neurons = (Neuron*) nullptr;
  input_neurons = (Neuron**) nullptr;
  output_neurons = (Neuron**) nullptr;

  combine_strings(ai_str_A, ai_str_B);
}


void NeuralWeb::from_string(const std::string& ai_str) {

  int current_char = 0;

  AmountOfNeurons = std::stoi(read_until(ai_str, current_char, ','));

  AmountOfInputs = std::stoi(read_until(ai_str, current_char, ','));

  AmountOfOutputs = std::stoi(read_until(ai_str, current_char, '|'));

  if (AmountOfNeurons < AmountOfInputs + AmountOfOutputs) {
    throw std::invalid_argument("Not enough neurons for the amount of inputs and outputs");
  }

  neurons = (Neuron*) malloc(AmountOfNeurons * sizeof(Neuron));

  // Generate Empty Neurons
  for (int neuron_i = 0; neuron_i < AmountOfNeurons; ++neuron_i) {
    neurons[neuron_i] = Neuron(neuron_i, 0);
  }

  int neuron_i = 0;

  while (ai_str[current_char] != '|') {

    if (neurons[neuron_i].id != std::stoi(read_until(ai_str, current_char, ':'))) { throw std::invalid_argument("File is not a valid NWeb"); }

    neurons[neuron_i].threshold = std::stof(read_until(ai_str, current_char, ';'));

    int downstream_conn_i = 0;

    while (ai_str[current_char] != '/') {

      neurons[neuron_i].downstream_connections[downstream_conn_i] = &neurons[std::stoi(read_until(ai_str, current_char, ','))];

      neurons[neuron_i].downstream_weights[downstream_conn_i] = std::stof(read_until(ai_str, current_char, '~'));

      downstream_conn_i++;

    }
    current_char++;

    neuron_i++;

  }

  // DOES NOT USE THE ACTUAL INPUT AND OUTPUT LIST:

  // Link input and output neurons
  input_neurons = (Neuron**) malloc(AmountOfInputs * sizeof(Neuron*));
  for (int input_neuron_i = 0; input_neuron_i < AmountOfInputs; ++input_neuron_i) {
    input_neurons[input_neuron_i] = &neurons[input_neuron_i];
  }

  output_neurons = (Neuron**) malloc(AmountOfOutputs * sizeof(Neuron*));
  for (int output_neuron_i = 0; output_neuron_i < AmountOfOutputs; ++output_neuron_i) {
    output_neurons[output_neuron_i] = &neurons[AmountOfInputs + output_neuron_i];
  }
}


void NeuralWeb::combine_strings(const std::string& ai_str_A, const std::string& ai_str_B) {

  int current_char_A = 0, current_char_B = 0;

  int temp_A = 0, temp_B = 0;

  temp_A = std::stoi(read_until(ai_str_A, current_char_A, ','));
  temp_B = std::stoi(read_until(ai_str_B, current_char_B, ','));
  if (temp_A == temp_B) {
    AmountOfNeurons = temp_A;
  } else {
    throw std::invalid_argument("Neural Webs are not the same size!");
  }

  temp_A = std::stoi(read_until(ai_str_A, current_char_A, ','));
  temp_B = std::stoi(read_until(ai_str_B, current_char_B, ','));
  if (temp_A == temp_B) {
    AmountOfInputs = temp_A;
  } else {
    throw std::invalid_argument("Neural Webs are not the same size!");
  }

  temp_A = std::stoi(read_until(ai_str_A, current_char_A, '|'));
  temp_B = std::stoi(read_until(ai_str_B, current_char_B, '|'));
  if (temp_A == temp_B) {
    AmountOfOutputs = temp_A;
  } else {
    throw std::invalid_argument("Neural Webs are not the same size!");
  }

  if (AmountOfNeurons < AmountOfInputs + AmountOfOutputs) {
    throw std::invalid_argument("Not enough neurons for the amount of inputs and outputs");
  }

  neurons = (Neuron*) malloc(AmountOfNeurons * sizeof(Neuron));

  // Generate Empty Neurons
  for (int neuron_i = 0; neuron_i < AmountOfNeurons; ++neuron_i) {
    neurons[neuron_i] = Neuron(neuron_i, 0);
  }

  int neuron_i = 0;

  std::uniform_real_distribution<float> random_parent(0.0f, 1.0f);

  while (ai_str_A[current_char_A] != '|' && ai_str_B[current_char_B] != '|') {
    if (neurons[neuron_i].id != std::stoi(read_until(ai_str_A, current_char_A, ':'))) { throw std::invalid_argument("File is not a valid NWeb"); }
    if (neurons[neuron_i].id != std::stoi(read_until(ai_str_B, current_char_B, ':'))) { throw std::invalid_argument("File is not a valid NWeb"); }

    if (random_parent(gen) <= 0.5) {

      neurons[neuron_i].threshold = std::stof(read_until(ai_str_A, current_char_A, ';'));
      read_until(ai_str_B, current_char_B, ';'); // <- skip other neuron

      int downstream_conn_i = 0;

      while (ai_str_A[current_char_A] != '/') {

        neurons[neuron_i].downstream_connections[downstream_conn_i] = &neurons[std::stoi(read_until(ai_str_A, current_char_A, ','))];

        neurons[neuron_i].downstream_weights[downstream_conn_i] = std::stof(read_until(ai_str_A, current_char_A, '~'));

        downstream_conn_i++;

      }

      // skip other neuron:
      while (ai_str_B[current_char_B] != '/') {
        read_until(ai_str_B, current_char_B, ',');
        read_until(ai_str_B, current_char_B, '~');

      }
      current_char_A++;
      current_char_B++;

      neuron_i++;

    } else {

      neurons[neuron_i].threshold = std::stof(read_until(ai_str_B, current_char_B, ';'));
      read_until(ai_str_A, current_char_A, ';'); // <- skip other neuron

      int downstream_conn_i = 0;

      while (ai_str_B[current_char_B] != '/') {

        neurons[neuron_i].downstream_connections[downstream_conn_i] = &neurons[std::stoi(read_until(ai_str_B, current_char_B, ','))];

        neurons[neuron_i].downstream_weights[downstream_conn_i] = std::stof(read_until(ai_str_B, current_char_B, '~'));

        downstream_conn_i++;

      }

      // skip other neuron:
      while (ai_str_A[current_char_A] != '/') {
        read_until(ai_str_A, current_char_A, ',');
        read_until(ai_str_A, current_char_A, '~');

      }
      current_char_A++;
      current_char_B++;

      neuron_i++;
    }



  }

  // DOES NOT USE THE ACTUAL INPUT AND OUTPUT LIST:

  // Link input and output neurons
  input_neurons = (Neuron**) malloc(AmountOfInputs * sizeof(Neuron*));
  for (int input_neuron_i = 0; input_neuron_i < AmountOfInputs; ++input_neuron_i) {
    input_neurons[input_neuron_i] = &neurons[input_neuron_i];
  }

  output_neurons = (Neuron**) malloc(AmountOfOutputs * sizeof(Neuron*));
  for (int output_neuron_i = 0; output_neuron_i < AmountOfOutputs; ++output_neuron_i) {
    output_neurons[output_neuron_i] = &neurons[AmountOfInputs + output_neuron_i];
  }
}

NeuralWeb::~NeuralWeb() {
  free(neurons);
  free(input_neurons);
  free(output_neurons);
}

/**
 * Run web for max amount of cycles.
 * Takes a pointer to an array of inputs of size AmountOfInputs and to an array of outputs of size AmountOfOutputs.
 * The current output of the neural web after the amount of cycles is written to the output array.
 * It returns the amount of cycles ran.
 */
int NeuralWeb::run_web(float *inputs, float *outputs, int max_cycles) {
  for (int input_neuron_i = 0; input_neuron_i < AmountOfInputs; ++input_neuron_i) {
    if (inputs[input_neuron_i] > 0.0f) {
      //input_neurons[input_neuron_i]->value -= input_neurons[input_neuron_i]->threshold;
      queue.emplace(input_neurons[input_neuron_i], inputs[input_neuron_i]);
    }
  }

  int cycle_i = 0;
  while (cycle_i < max_cycles) {
    ++cycle_i;
    //printf("Head: %d  -  Tail: %d  -  Used Queue size: %d\n",
    //       neuron_queue.current_head, neuron_queue.current_tail,
    //       neuron_queue.current_head - neuron_queue.current_tail);

    Neuron *current_neuron = queue.front().neuron;

    current_neuron->update(queue.front().sent_value);

    if (current_neuron->value > current_neuron->threshold) {
      current_neuron->value -= current_neuron->threshold;
      for (int neuron_i = 0; neuron_i < NEURON_DOWNSTREAM_SIZE; ++neuron_i) {
        if (current_neuron->downstream_connections[neuron_i] != nullptr) {
          queue.emplace(current_neuron->downstream_connections[neuron_i],
                        current_neuron->value * current_neuron->downstream_weights[neuron_i]);
        }
      }
    }

    queue.pop();

    if (queue.empty()) {
      printf("Neural web is braindead...\n");
      break;
    }

    if (cycle_i % 100 == 0) {
      printf("Cycle: %d, Queue size: %zu\n", cycle_i, queue.size());
    }

  }

  for (int output_neuron_i = 0; output_neuron_i < AmountOfOutputs; ++output_neuron_i) {
    outputs[output_neuron_i] = neurons[AmountOfInputs + output_neuron_i].value;
  }

  return cycle_i;
}

void NeuralWeb::to_json(const std::string& filename,
                        const std::filesystem::path& dirPath) {
  if (!std::filesystem::exists(dirPath)) {
    std::filesystem::create_directory(dirPath);
  }

  std::string json;

  json += "{\n";

  json += "  \"amount_of_neurons\" : " + std::to_string(AmountOfNeurons) + ",\n";
  json += "  \"amount_of_inputs\" : " + std::to_string(AmountOfInputs) + ",\n";
  json += "  \"amount_of_outputs\" : " + std::to_string(AmountOfOutputs) + ",\n";

  json += "  \"neurons\" : {";

  for (int neuron_i = 0; neuron_i < AmountOfNeurons; ++neuron_i) {
    json += "\n    \"" + std::to_string(neurons[neuron_i].id) + "\" : {\n";
    json += "      \"threshold\" : " + std::to_string(neurons[neuron_i].threshold) + ",\n";
    json += "      \"downstream\" : [";
    for (int downstream_i = 0; downstream_i < NEURON_DOWNSTREAM_SIZE; ++downstream_i) {
      if (neurons[neuron_i].downstream_connections[downstream_i] != nullptr) {
        json += "\n        {\n";
        json += "          \"neuron\" : "
                + std::to_string(neurons[neuron_i].downstream_connections[downstream_i]->id)
                + ",\n";
        json += "          \"weight\" : "
                + std::to_string(neurons[neuron_i].downstream_weights[downstream_i])
                + "\n";
        json += "        },";
      }
    }
    json.pop_back(); // <- remove trailing comma
    json += "\n";
    json += "      ]\n"
            "    },";
  }

  json.pop_back(); // <- remove trailing comma
  json += "\n  },\n  \"inputs\" : [";
  for (int input_neuron_i = 0; input_neuron_i < AmountOfInputs; ++input_neuron_i) {
    json += " " + std::to_string(input_neurons[input_neuron_i]->id) + ",";
  }
  json.pop_back(); // <- remove trailing comma
  json += " ],\n";

  json += "  \"outputs\" : [";
  for (int output_neuron_i = 0; output_neuron_i < AmountOfOutputs; ++output_neuron_i) {
    json += " " + std::to_string(output_neurons[output_neuron_i]->id) + ",";
  }
  json.pop_back(); // <- remove trailing comma
  json += " ]\n";

  json += "}";

  std::ofstream file(dirPath.string() + "/" + filename);
  file << json;
  file.close();
}

void NeuralWeb::to_string(const std::string& filename,
                          const std::filesystem::path& dirPath) {
  if (!std::filesystem::exists(dirPath)) {
    std::filesystem::create_directory(dirPath);
  }

  std::ofstream file(dirPath.string() + "/" + filename);
  file << to_string();
  file.close();
}


std::string NeuralWeb::to_string() {

  std::string str;

  str += std::to_string(AmountOfNeurons) + ",";
  str += std::to_string(AmountOfInputs) + ",";
  str += std::to_string(AmountOfOutputs);

  str += "|";

  for (int neuron_i = 0; neuron_i < AmountOfNeurons; ++neuron_i) {
    str += std::to_string(neurons[neuron_i].id) + ":";
    str += std::to_string(neurons[neuron_i].threshold) + ";";
    for (int downstream_i = 0; downstream_i < NEURON_DOWNSTREAM_SIZE; ++downstream_i) {
      if (neurons[neuron_i].downstream_connections[downstream_i] != nullptr) {
        str += std::to_string(neurons[neuron_i].downstream_connections[downstream_i]->id) + ",";
        str += std::to_string(neurons[neuron_i].downstream_weights[downstream_i]) + "~";
      }
    }
    str += "/";
  }

  str += "|";
  for (int input_neuron_i = 0; input_neuron_i < AmountOfInputs; ++input_neuron_i) {
    str += std::to_string(input_neurons[input_neuron_i]->id) + ",";
  }
  str += ";";

  for (int output_neuron_i = 0; output_neuron_i < AmountOfOutputs; ++output_neuron_i) {
    str += std::to_string(output_neurons[output_neuron_i]->id) + ",";
  }
  str += "?";

  return str;
}
