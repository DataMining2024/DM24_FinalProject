import numpy as np
from src.data import Dataset
from src.train import Training
from src.model import Chemomile

parameters = dict(
    target = 'HCOM',
    subfrag_size = 12,
    edge_size = 3,
    out_size = 1,
    seed = 42,
    batch_size = 256,
    max_epoch = 200,
    verbose = False,
    save = False,
)

# hidden_size, dropout, num_layers, num_timestep, lr_init, gamma, weight_decay
bounds = (np.array([ 10, 0.2, 1, 1, 1, 0.975, 0.0]),  # lower bounds
          np.array([100, 0.5, 5, 5, 5, 0.999, 0.1]))  # upper bounds

dataset = Dataset(
    target = parameters['target'],
    seed = parameters['seed'],
    batch_size = parameters['batch_size'],
    verbose = parameters['verbose']
)

def target_function(input_array):
    new_params = parameters
    input_array = input_array.flatten()
    
    new_params['hidden_size'] = int(input_array[0])
    new_params['dropout'] = float(input_array[1])
    new_params['num_layers'] = int(input_array[2])
    new_params['num_timesteps'] = int(input_array[3])
    new_params['lr_init'] = float(pow(10, -1 * input_array[4]))
    new_params['gamma'] = float(input_array[5])
    new_params['weight_decay'] = float(input_array[6])

    model = Chemomile(
            subfrag_size = new_params['subfrag_size'],
            hidden_size = new_params['hidden_size'],
            out_size = new_params['out_size'],
            edge_size = new_params['edge_size'],
            dropout = new_params['dropout'],
            num_layers = new_params['num_layers'],
            num_timesteps = new_params['num_timesteps']
        )
    
    train = Training(model, new_params, dataset = dataset)
    train.run()

    print(f"\nTestLoss : {train.test_loss}, RMSE : {train.rmse}")

    return np.array([train.test_loss])

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles = 10, dimensions = 7,
                                    bounds = bounds, options = options)

stats = optimizer.optimize(target_function, iters = 50, n_processes = 10)

print(stats)

