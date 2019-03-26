from learn.neural_net import neural_net


#################
################# NEURAL NETWORK FACTORY
#################


class neural_factory:

    def get_neural_net(no_of_inputs, no_of_neurons_layer1, no_of_neurons_layer2,
                   learning_rate=0.0001, network_name="deep_net"):

        deep_network = neural_net(no_of_inputs=no_of_inputs, 
                                     no_of_neurons_layer1=no_of_neurons_layer1, 
                                     no_of_neurons_layer2=no_of_neurons_layer2, \
                                     learning_rate=learning_rate, 
                                     network_name=network_name)
        return deep_network