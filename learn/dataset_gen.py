import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split

from learn.matrix_init import matrix_init
from learn.neural_layer import neural_layer



#################
################# DATASET GENERATION
#################

class dataset_gen:
    
    # create a dataset
    def get_test_dataset():
        dataset_size =10
        random_bias_set = matrix_init.truncated_normal(mean=1, sd=10, low=1, upp=10).rvs(dataset_size)
        random_input_set = matrix_init.truncated_normal(mean=10, sd=100, low=1, upp=100).rvs(dataset_size)
        input_examples = np.column_stack((random_bias_set, random_input_set))
        # print("input_examples=", input_examples, input_examples.shape)
        # print("random_bias_set[0]", random_bias_set[0], " random_input_set[0]", random_input_set[0])

        line_slope = 2
        output_training_set = random_bias_set + line_slope*random_input_set
        # print("output_training_set=", output_training_set, output_training_set.shape)

        training_dataset = np.column_stack((random_input_set, output_training_set))

        return random_input_set, input_examples, output_training_set, training_dataset

    # train model: neural_layer
    def run_training(random_input_set, input_examples, output_training_set, training_dataset): 
        learning_rate = 0.0001
        neural_lineup = neural_layer(no_of_inputs=2, no_of_neurons=1, learning_rate=learning_rate, \
                                    layer_name="least_squares_network")

        no_of_epochs = 10
        neuron_losses = np.zeros((no_of_epochs, 2))
        for i in range(no_of_epochs):
            neuron_losses[i, 0] = i
            predicted_outputs, input_losses = neural_lineup.train(input_examples, output_training_set)
            neuron_losses[i, 1] = LA.norm(input_losses)

        return neural_lineup, neuron_losses

#     random_input_set, input_examples, output_training_set, training_dataset = get_test_dataset()
#     neuron_x, neuron_x_losses = run_training(random_input_set, input_examples, output_training_set, training_dataset)

    def get_singular_value_decomposition(network_layer):
        u_matrix, sigma_matrix, v_matrix_transpose = \
            LA.svd(network_layer.input_parameters, full_matrices=True)
        # print("layer1: u_sine=", u_sine)
        # print("layer1: sigma_sine=", sigma_sine)
        # print("layer1: v_sine_transpose=", v_sine_transpose)
        return u_matrix, sigma_matrix, v_matrix_transpose
    
    def get_singular_value_range(network_layer):
        u_matrix, sigma_matrix, v_matrix_transpose = dataset_gen.get_singular_value_decomposition(network_layer)
        return sigma_matrix[0], sigma_matrix[len(sigma_matrix)-1]

    def data_default_gathering(input_set, output_set, neural_net, no_of_epochs):
        # defaults
        dropout_rate = 0
        training_stochasticity_rate = 1
        return dataset_gen.data_gathering(input_set, output_set, neural_net, no_of_epochs, \
                              dropout_rate, training_stochasticity_rate)
    
    def init_matrix(num_rows):
        return np.zeros((num_rows, 2))
    
    def calculate_test_error(input_test_set, output_test_set, neural_net):
        total_error = 0
        # propagate
        dropout_rate = 0.0
        training_outputs_layer1 = neural_net.layer1.forward_dropping_propagate(input_test_set, dropout_rate)
        predicted_outputs_layer2, losses_backpropagated_layer2 = \
            neural_net.layer2.train(training_outputs_layer1, output_test_set)
        for prediction_index in range(len(predicted_outputs_layer2)):
            predicted_values = predicted_outputs_layer2[prediction_index]
            actual_values = output_test_set[prediction_index]
            actual_values = [float(output_test_set[prediction_index][0]), float(output_test_set[prediction_index][1])]
            #print("predicted_thing=", predicted_values)
            #print("actual_thing=", actual_values)
            delta_errors = np.subtract(predicted_values, actual_values)
            total_error += LA.norm(delta_errors)
            #print("delta_error=", delta_errors, " total_error=", total_error)
        return total_error
            

    def data_gathering(input_set, output_set, neural_net, \
                       no_of_epochs, \
                       dropout_rate, training_stochasticity_rate):
        
        input_training_set, input_test_set, output_training_set, output_test_set = \
            train_test_split(input_set, output_set, test_size=0.33, random_state=42)
                
        # to save data during training
        network_losses_layer2 = dataset_gen.init_matrix(no_of_epochs)
        network_losses_layer1 = dataset_gen.init_matrix(no_of_epochs)
        network_losses_layer0 = dataset_gen.init_matrix(no_of_epochs)
        param_conditioning_layer2 = dataset_gen.init_matrix(no_of_epochs)
        param_conditioning_layer1 = dataset_gen.init_matrix(no_of_epochs)
        large_singular_value_layer1 = dataset_gen.init_matrix(no_of_epochs)
        small_singular_value_layer1 = dataset_gen.init_matrix(no_of_epochs)
        large_singular_value_layer2 = dataset_gen.init_matrix(no_of_epochs)
        small_singular_value_layer2 = dataset_gen.init_matrix(no_of_epochs)
        error_for_conditioning = dataset_gen.init_matrix(no_of_epochs)
        error_gain_layer2 = dataset_gen.init_matrix(no_of_epochs)

        for i in range(no_of_epochs):
            # key
            network_losses_layer2[i, 0] = i
            network_losses_layer1[i, 0] = i
            network_losses_layer0[i, 0] = i
            param_conditioning_layer2[i, 0] = i
            param_conditioning_layer1[i, 0] = i
            large_singular_value_layer1[i, 0] = i
            small_singular_value_layer1[i, 0] = i
            large_singular_value_layer2[i, 0] = i
            small_singular_value_layer2[i, 0] = i
            error_for_conditioning[i, 0] = i 
            error_gain_layer2[i, 0] = i
            # train
            predicted_outputs_layer2, loss_layer2, losses_backpropagated_layer2, losses_backpropagated_layer1 = \
                neural_net.train(input_training_set, output_training_set, dropout_rate, training_stochasticity_rate)
            # value
            network_losses_layer2[i, 1] = loss_layer2
            network_losses_layer1[i, 1] = LA.norm(losses_backpropagated_layer2)
            network_losses_layer0[i, 1] = LA.norm(losses_backpropagated_layer1)
            param_conditioning_layer2[i, 1] = LA.cond(neural_net.layer2.input_parameters)
            param_conditioning_layer1[i, 1] = LA.cond(neural_net.layer1.input_parameters)
            large_singular_value_layer1[i, 1], small_singular_value_layer1[i, 1] = \
                dataset_gen.get_singular_value_range(neural_net.layer1)
            large_singular_value_layer2[i, 1], small_singular_value_layer2[i, 1] = \
                dataset_gen.get_singular_value_range(neural_net.layer2)
            # conditioning vs total error: evaluate entire test set at every epoch
            weight_conditioning = param_conditioning_layer2[i, 1]
            total_error = dataset_gen.calculate_test_error(input_test_set, output_test_set, neural_net)
            error_for_conditioning[i, 1] = total_error # plot for weight_conditioning
            try:
                error_gain_layer2[i, 1] = abs(error_for_conditioning[i, 1] - error_for_conditioning[i-1, 1])
            except:
                error_gain_layer2[i, 1] = 0
                print("first_element_error")
            
        return predicted_outputs_layer2, network_losses_layer2, \
            network_losses_layer1, network_losses_layer0, \
            param_conditioning_layer2, param_conditioning_layer1, \
            large_singular_value_layer1, small_singular_value_layer1, \
            large_singular_value_layer2, small_singular_value_layer2, \
            error_for_conditioning, error_gain_layer2 