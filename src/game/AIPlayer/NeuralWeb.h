#ifndef FASTCATAN_NEURALWEB_H
#define FASTCATAN_NEURALWEB_H

class Neuron {

  explicit Neuron(int downstream_size);
  ~Neuron();

  int value;

  int amount_of_downstream_connections;

  void update(int in_value);

  Neuron *downstream_connections;
  int *downstream_weights;

};


class NeuralWeb {

  NeuralWeb(int amount_of_neurons, int amount_of_inputs, int amount_of_outputs);

  Neuron *neurons;

  Neuron  *input_neurons;
  Neuron *output_neurons;

  int neuron_queue_size = 200;
  int neuron_downstream_size = 10;

  void run_web(int cycles);

  ~NeuralWeb();


};


#endif //FASTCATAN_NEURALWEB_H
