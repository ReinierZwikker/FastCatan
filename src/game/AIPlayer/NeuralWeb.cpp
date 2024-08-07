#include <cstdlib>
#include "NeuralWeb.h"

/******************
 *     NEURON     *
 ******************/

Neuron::Neuron(int downstream_size) {
  value = 0;

  downstream_connections = (Neuron*) malloc(downstream_size * sizeof(Neuron*));
  downstream_weights = (int*) malloc(downstream_size * sizeof(int))
}

Neuron::~Neuron() {

}

void Neuron::update(int in_value) {
  value += in_value;
  for (int conn_i = 0; conn_i < amount_of_downstream_connections; ++conn_i) {
    downstream_connections[conn_i].update(value * downstream_weights[conn_i]);
  }
}


/******************
 *      WEB       *
 ******************/

NeuralWeb::NeuralWeb(int amount_of_neurons, int amount_of_inputs, int amount_of_outputs) {

}

NeuralWeb::~NeuralWeb() {

}

void NeuralWeb::run_web(int cycles) {

}
