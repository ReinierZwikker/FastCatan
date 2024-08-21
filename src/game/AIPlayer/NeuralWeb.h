#ifndef FASTCATAN_NEURALWEB_H
#define FASTCATAN_NEURALWEB_H

#define NEURON_QUEUE_SIZE 200
#define NEURON_DOWNSTREAM_SIZE 10

class Neuron {

  explicit Neuron(Neuron *set_downstream_connections[NEURON_DOWNSTREAM_SIZE],
                  float set_downstream_weights[NEURON_DOWNSTREAM_SIZE],
                  float set_threshold);

public:
  float value, threshold;

  void update(float in_value);

  Neuron *downstream_connections[NEURON_DOWNSTREAM_SIZE] {};
  float downstream_weights[NEURON_DOWNSTREAM_SIZE] {};
};


struct QueuedNeuron {
    inline QueuedNeuron(Neuron *neuron_to_queue, float value_to_send)
             { neuron = neuron_to_queue; sent_value = value_to_send; }

    Neuron *neuron;
    float sent_value;
};

class NeuronQueue {
    ~NeuronQueue();

    QueuedNeuron *queue;

    int current_head = 0;
    int current_tail = 0;

public:
    NeuronQueue();

    void run_next_neuron();

    void add_to_queue(Neuron *neuron, float value);
};


class NeuralWeb {

  NeuralWeb(int amount_of_neurons,
            int amount_of_inputs,
            int amount_of_outputs);

  Neuron *neurons;

  Neuron  *input_neurons;
  Neuron *output_neurons;

  NeuronQueue neuron_queue;

  int AmountOfNeurons,
      AmountOfInputs,
      AmountOfOutputs;

  void run_web(float *inputs, int max_cycles);

  ~NeuralWeb();


};


#endif //FASTCATAN_NEURALWEB_H
