#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include "NeuralWeb.h"

/******************
 *     NEURON     *
 ******************/

Neuron::Neuron(Neuron *set_downstream_connections[NEURON_DOWNSTREAM_SIZE],
               float set_downstream_weights[NEURON_DOWNSTREAM_SIZE],
               float set_threshold) {
  value = 0;
  threshold = set_threshold;


  for (int neuron_i = 0; neuron_i < NEURON_DOWNSTREAM_SIZE; ++neuron_i) {
    downstream_connections[neuron_i] = set_downstream_connections[neuron_i];
    downstream_weights[neuron_i] = set_downstream_weights[neuron_i];
  }
}

void Neuron::update(float in_value) {
  value += in_value;
}


/******************
 *     QUEUE      *
 ******************/

NeuronQueue::NeuronQueue() {
  queue = (QueuedNeuron*) malloc(NEURON_QUEUE_SIZE * sizeof(QueuedNeuron));

  for (int neuron_i = 0; neuron_i < NEURON_QUEUE_SIZE; ++neuron_i) {
    queue[neuron_i] = QueuedNeuron(nullptr, 0);
  }
}

NeuronQueue::~NeuronQueue() {
  free(queue);
}

void NeuronQueue::run_next_neuron() {
  Neuron *current_neuron = queue[current_tail].neuron;

  current_neuron->update(queue[current_tail].sent_value);

  if (current_neuron->value > current_neuron->threshold) {
    for (int neuron_i = 0; neuron_i < NEURON_DOWNSTREAM_SIZE; ++neuron_i) {
      if (current_neuron->downstream_connections[neuron_i] != nullptr) {
        add_to_queue(current_neuron->downstream_connections[neuron_i],
                     current_neuron->value * current_neuron->downstream_weights[neuron_i]);
      }
    }
  }

  ++current_tail;
  if (current_tail >= NEURON_QUEUE_SIZE) { current_tail = 0; }
}

void NeuronQueue::add_to_queue(Neuron *neuron, float value) {
  ++current_head;
  if (current_head >= NEURON_QUEUE_SIZE) { current_head = 0; }
  if (current_head == current_tail) { printf("The snake ate its tail!"); }

  queue[current_head] = QueuedNeuron(neuron, value);
}


/******************
 *      WEB       *
 ******************/

NeuralWeb::NeuralWeb(int amount_of_neurons,
                     int amount_of_inputs,
                     int amount_of_outputs) {
  AmountOfNeurons = amount_of_neurons;
  AmountOfInputs = amount_of_inputs;
  AmountOfOutputs = amount_of_outputs;

  if (AmountOfNeurons < AmountOfInputs + AmountOfOutputs) {
    throw std::invalid_argument("Not enough neurons for the amount of inputs and outputs");
  }

  neurons = (Neuron*) malloc(AmountOfNeurons * sizeof(Neuron));

  input_neurons = (Neuron*) malloc(AmountOfInputs * sizeof(Neuron*));
  for (int input_neuron_i = 0; input_neuron_i < AmountOfInputs; ++input_neuron_i) {
    input_neurons[input_neuron_i] = neurons[input_neuron_i];
  }

  output_neurons = (Neuron*) malloc(AmountOfOutputs * sizeof(Neuron*));
  for (int output_neuron_i = 0; output_neuron_i < AmountOfOutputs; ++output_neuron_i) {
    output_neurons[output_neuron_i] = neurons[AmountOfInputs + output_neuron_i];
  }

  

}

NeuralWeb::~NeuralWeb() {

}

void NeuralWeb::run_web(float *inputs, int max_cycles) {
  for (int input_neuron_i = 0; input_neuron_i < AmountOfInputs; ++input_neuron_i) {
    neuron_queue.add_to_queue(&input_neurons[input_neuron_i], inputs[input_neuron_i]);
  }

  int cycle_i = 0;
  while (cycle_i < max_cycles) {
    ++cycle_i;
    neuron_queue.run_next_neuron();
  }

}
