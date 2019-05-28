import numpy as np

import csv

class file_io:
    
    def load_dataset(file_name):
        with open(file_name, 'r') as csvfile:
            file_reader = csv.DictReader(csvfile)
            data_set = {}
            for file_row in file_reader:
                row_price = file_row['price']
                row_reward = file_row['reward']
                data_set[row_price] = row_reward
            return data_set

    def get_dataset(dataset_name):
        accumulate_filename = dataset_name + "_accumulate_q_table.csv"
        distribute_filename = dataset_name + "_distribute_q_table.csv"
        accumulate_data = file_io.load_dataset(accumulate_filename)
        distribute_data = file_io.load_dataset(distribute_filename)
        print(accumulate_filename, " length=", len(accumulate_data))
        print(distribute_filename, " length=", len(distribute_data))
        dataset_inputs = []
        dataset_outputs = []
        for key, accumulate_value in accumulate_data.items():
            try:
                distribute_value = distribute_data[key] 
                next_outputs = [accumulate_value, distribute_value]
                if(len(dataset_inputs)==0):
                    dataset_inputs = [key]
                    dataset_outputs = next_outputs
                    # print("initial_dataset_inputs=", dataset_inputs)
                    # print("initial_dataset_outputs=", dataset_outputs)
                else:
                    dataset_inputs = np.vstack((dataset_inputs, [key])) # dataset_inputs + [key]
                    dataset_outputs = np.vstack((dataset_outputs, next_outputs))
                    pass
            except Exception as e:
                # print("unmatched_key=", key, str(e))
                pass
        print("dataset_inputs[0]=", dataset_inputs[0], len(dataset_inputs))
        print("dataset_outputs[0]=", dataset_outputs[0], dataset_outputs.shape)
        return dataset_inputs, dataset_outputs
