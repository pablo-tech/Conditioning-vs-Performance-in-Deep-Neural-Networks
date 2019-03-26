# NEURAL LAYER: used to construct neural networks
import numpy as np
from numpy import linalg as LA
import random
from scipy.stats import truncnorm

from learn.matrix_init import matrix_init


#################
################# NEURAL LAYER
#################

class neural_layer:
    def __init__(self, no_of_inputs, no_of_neurons, learning_rate, layer_name):
        self.no_of_inputs = no_of_inputs
        print(layer_name, " no_of_inputs=", no_of_inputs)
        self.no_of_neurons = no_of_neurons
        print(layer_name, " no_of_neurons=", no_of_neurons)
        self.learning_rate = learning_rate
        print(layer_name, "layer_name learning_rate=", learning_rate)
        self.layer_name = layer_name

        self.input_parameters = self.create_weight_matrix()

        
    def create_weight_matrix(self):
        no_of_bias_inputs = 1
        return self.layer_parameters(self.no_of_neurons, no_of_bias_inputs + self.no_of_inputs)

    def absolute_range(self, num_inputs):
        return 1 / np.sqrt(self.no_of_inputs) # good range for initial params

    def parametrized_random(self, absolute_range):
        return matrix_init.truncated_normal(mean=2, sd=1, low=-absolute_range, upp=absolute_range)

    def layer_parameters(self, no_of_inputs, no_of_neurons):  # fully connected
        layer_dimensions = (no_of_neurons, no_of_inputs)
        absolute_range = self.absolute_range(layer_dimensions)
        layer_weigths_matrix = self.parametrized_random(absolute_range).rvs(layer_dimensions)
        print(self.layer_name, " weigths_matrix=", layer_weigths_matrix, layer_weigths_matrix.shape)
        return layer_weigths_matrix

    def append_bias(self, input_vectors):
        # m = len(input_vectors)       # 
        m,n = input_vectors.shape  # m,n 
        bias_column = np.ones((m,1)) # x0=1 for bias
        biased_inputs = np.hstack((bias_column, input_vectors))
        return biased_inputs  

    def forward_propagate(self, input_vectors):
        dropout_rate = 0.0 # keep value 100% of time
        return self.forward_dropping_propagate(input_vectors, dropout_rate)

    # matrix*vector per neuron => matrix*matrix for layer
    def forward_dropping_propagate(self, input_vectors, dropout_rate):            
        # print(self.layer_name, " input_vectors=", input_vectors, input_vectors.shape)
        input_vectors = self.append_bias(input_vectors)
        # print(self.layer_name, " biased_input_set=", input_vectors, input_vectors.shape)
        params_to_use = self.input_parameters.copy()
        for i in range(self.no_of_neurons):
            if(random.uniform(0, 1) < dropout_rate):
                m,n = params_to_use.shape
                for j in range(m):
                    params_to_use[j, i] = 0        
        # print("params_to_use=", params_to_use)
        
        input_vectors = input_vectors.astype(float) # map(float, input_vectors)
        params_to_use = params_to_use.astype(float) # map(float, params_to_use)
        layer_training_outputs = np.dot(input_vectors, params_to_use) 
        # print(self.layer_name, " predicted_output_set=", layer_training_outputs.shape)
        return layer_training_outputs

    def compute_loss(self, predicted_output_set, true_output_set):
        predicted_output_set = np.array(predicted_output_set)
        true_output_set = np.array(true_output_set)
        #print("true_output_set=", true_output_set.shape)
        #print("predicted_output_set=", predicted_output_set.shape)
        layer_residuals = true_output_set.astype(float) - predicted_output_set.astype(float)
        # print(self.layer_name, " layer_residuals=", layer_residuals, layer_residuals.shape)
        layer_loss = LA.norm(layer_residuals) 
        return layer_loss

    # all examples, whole layer
    def train(self, input_examples, true_output):
        predicted_outputs = self.forward_propagate(input_examples)  
        # print("==> layer_predicted_outputs[0]=", predicted_outputs[0], predicted_outputs.shape)
        # print("==> layer_true_outputs[0]=", true_output[0], true_output.shape)
        backprop_input_loss_total = np.zeros(self.no_of_inputs)
        for i in range(self.no_of_neurons):
            if self.no_of_neurons==1:
                neuron_predictions = predicted_outputs 
                neuron_truth = np.array(true_output)
            else:
                neuron_predictions = [row[i] for row in predicted_outputs] 
                neuron_truth = [row[i] for row in true_output] 
            #print("neuron_predictions_1=", neuron_predictions)
            #print("neuron_truth_1=", neuron_truth)
            backprop_input_loss = self.train_neuron(i, input_examples, neuron_truth, neuron_predictions)
            backprop_input_loss_total += backprop_input_loss 
        return predicted_outputs, backprop_input_loss_total 
            
    def neuron_params(self, neuron_index):
        return [row[neuron_index] for row in self.input_parameters]
    
    # all examples, single neuron in the layer        
    # bias & weights: change simultaneously, having considered loss for all training examples in a single neuron
    def train_neuron(self, neuron_index, input_examples, true_outputs, neuron_predictions):
        # print("==> neuron_predicted_outputs[0]=", neuron_predictions[0], np.array(neuron_predictions).shape) 
        # print("==> neuron_true_outputs[0]=", true_outputs[0], np.array(true_outputs).shape)
        examples_loss = self.compute_loss(neuron_predictions, true_outputs)
        # print(self.layer_name, " examples_loss=", examples_loss)
        backprop_input_loss = \
            self.train_params(input_examples, true_outputs, neuron_predictions, neuron_index)
        # print(self.layer_name, " trained_parameters=", self.neuron_params(neuron_index))
        # print(self.layer_name, " backprop_input_loss=", backprop_input_loss)
        return backprop_input_loss
            
    def train_params(self, input_examples, true_outputs, predicted_outputs, neuron_index):
        input_params = self.neuron_params(neuron_index)
        # print(self.layer_name, " input_examples=", input_examples)
        # print(self.layer_name, " input_params=", input_params)
        self.input_parameters[0, neuron_index] = \
            self.update_neuron_bias(input_examples, true_outputs, predicted_outputs, input_params)
        self.input_parameters[1:, neuron_index], back_propagated_losses = \
            self.update_neuron_weights(input_examples, true_outputs, predicted_outputs, input_params)
        # print("params_after_saved=", self.input_parameters[1:, neuron_index])
        return back_propagated_losses

    def update_neuron_bias(self, input_examples, true_output, predicted_outputs, input_params): 
        # print("...UPDATE_BIAS:")
        bias_index = 0
        param_before = input_params[bias_index] # just bias
        no_of_examples = len(input_examples)
        cummultive_loss = 0
        for example_index in range(no_of_examples):  
            example_loss =  predicted_outputs[example_index].astype(float) - \
                true_output[example_index].astype(float)
            # print("example_loss=", example_loss)
            cummultive_loss += example_loss
        param_after = param_before - self.learning_rate * cummultive_loss / no_of_examples
        print("*** FINAL_BIAS=", param_after, param_after.shape)
        return param_after
            
    def update_neuron_weights(self, input_examples, true_outputs, predicted_outputs, input_params):
        params_before = input_params[1:] # all rows, all columns except bias
        params_after = np.zeros(len(params_before))
        # print("... weight_params=", params_after, params_after.shape)
        back_propagated_losses = np.zeros(self.no_of_inputs)
        for input_index in range(self.no_of_inputs):
            param_before = params_before[input_index]
            no_of_examples = len(input_examples)
            cummultive_loss = 0
            for example_index in range(no_of_examples):
                # print("...UPDATE_WEIGHT:", " input_index=", input_index, " example_index=", example_index)
                example_residual = predicted_outputs[example_index].astype(float) - \
                    true_outputs[example_index].astype(float)  # [0]
                input_column = input_examples[example_index][input_index]
                example_loss =  example_residual * input_column 
                # print("example_loss=", example_loss)
                cummultive_loss += example_loss
                back_propagated_losses[input_index] += LA.norm([example_residual]) / input_column
            loss_per_example = cummultive_loss / no_of_examples    
            params_after[input_index] = param_before - self.learning_rate * loss_per_example
        print("*** FINAL_WEIGHTS[0]=", params_after[0], params_after.shape)
        print("*** BACKPROP[0] -> ", back_propagated_losses[0], back_propagated_losses.shape)
        return params_after, back_propagated_losses
                