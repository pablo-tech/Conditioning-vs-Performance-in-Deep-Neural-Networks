import numpy as np
from numpy import linalg as LA
import random

from learn.neural_layer import neural_layer


#################
################# NEURAL NETWORK
#################

class neural_net:
    
    def __init__(self, no_of_inputs, no_of_neurons_layer1, no_of_neurons_layer2, 
                 learning_rate, network_name):
        self.no_of_inputs = no_of_inputs
        self.layer1 = neural_layer(no_of_inputs, no_of_neurons_layer1, learning_rate, network_name + "&layer1")
        self.layer2 = neural_layer(no_of_neurons_layer1, no_of_neurons_layer2, learning_rate, network_name + "&layer2")
        self.learning_rate = learning_rate
        self.network_name = network_name
                    
    def append_bias(self, input_vector):
        return np.append([1], input_vector) # first element is 1 for bias 
    
    def backward_propagate(self, predicted_output, true_output):
        print("propagate loss")
    
    def train_default(self, input_examples, true_output):
        # defaults 
        dropout_rate = 0
        training_stochasticity_rate = 1
        return self.train(self, input_examples, true_output, dropout_rate, training_stochasticity_rate)
    
    def train(self, input_examples, true_output, dropout_rate, training_stochasticity_rate):
        # print(self.network_name, " TRAINING_input[0]=>", input_examples[0], input_examples.shape)
        # print(self.network_name, " TRAINING_truth[0]=>", true_output[0], true_output.shape)
        training_outputs_layer1 = self.layer1.forward_dropping_propagate(input_examples, dropout_rate)
        predicted_outputs_layer2, losses_backpropagated_layer2 = \
            self.layer2.train(training_outputs_layer1, true_output)
        loss_layer2 = self.layer2.compute_loss(predicted_outputs_layer2, true_output)
        
        losses_backpropagated_layer1 = np.zeros(self.no_of_inputs)
        for neuron_index in range(self.layer1.no_of_neurons):
            no_of_training_examples = len(input_examples)
            for training_example in input_examples:
                if(random.uniform(0, 1) > training_stochasticity_rate): 
                    neuron_params = self.layer1.neuron_params(neuron_index)
                    bias_index = 0
                    self.layer1.neuron_params(neuron_index)[bias_index] -= \
                        losses_backpropagated_layer2 / no_of_training_examples
                    for param_index in range(1, self.layer1.no_of_inputs):
                        self.layer1.neuron_params(neuron_index)[param_index] = \
                            losses_backpropagated_layer2 / no_of_training_examples / training_example[param_index]
                        losses_backpropagated_layer1[param_index] += LA.norm(losses_backpropagated_layer2) / \
                            self.layer1.neuron_params(neuron_index)[param_index]
        return predicted_outputs_layer2, loss_layer2, losses_backpropagated_layer2, losses_backpropagated_layer1
