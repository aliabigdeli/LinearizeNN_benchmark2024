import sys
import h5py
import torch
import torch.nn as nn
import numpy as np
import os
import onnx
import warnings
import struct
from typing import List

data_type_tab = {
    1: ['f', 4],
    2: ['B', 1],
    3: ['b', 1],
    4: ['H', 2],
    5: ['h', 2],
    6: ['i', 4],
    7: ['q', 8],
    10: ['e', 2],
    11: ['d', 8],
    12: ['I', 4],
    13: ['Q', 8]
}

def unpack_weights(initializer):
    ret = {}
    for i in initializer:
        name = i.name
        dtype = i.data_type
        shape = list(i.dims)
        if dtype not in data_type_tab:
            warnings("This data type {} is not supported yet.".format(dtype))
        fmt, size = data_type_tab[dtype]
        if len(i.raw_data) == 0:
            if dtype == 1:
                data_list = i.float_data
            elif dtype == 7:
                data_list = i.int64_data
            else:
                warnings.warn("No-raw-data type {} not supported yet.".format(dtype))
        else:
            data_list = struct.unpack('<' + fmt * (len(i.raw_data) // size), i.raw_data)
        t = torch.tensor(data_list)
        if len(shape) != 0:
            t = t.view(*shape)
        ret[name] = t
    return ret

class ONNXNet(nn.Module):
    def __init__(self):
        super(ONNXNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 8),
            nn.ReLU(),

            nn.Linear(8, 2),
        )
        self.controller = nn.Linear(2, 1, bias=False)
        self.linear_weight = nn.Linear(2, 1, bias=False)
        self.linear_constraint = nn.Linear(2, 2, bias=False)
    
    def forward(self, x):
        output1 = self.controller(self.main(x))
        output2 = self.linear_weight(x[:, 2:])
        concatenated = torch.cat((output1, output2), dim=1)
        output3 = self.linear_constraint(concatenated)
        return output3

def save_vnnlib(input_bounds: np.ndarray, output_bounds: np.ndarray, vnnlib_file_name: str):

    """
    Saves the property derived as vnn_lib format.
    """

    with open(vnnlib_file_name, "w") as f:

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(output_bounds.shape[0]):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write(f"(assert (or")
        f.write(f"\n")
        for i in range(output_bounds.shape[0]):
            f.write(f"    (and (>= Y_{i} {output_bounds[i]}))\n")
        f.write(f"))")
        f.write("\n")

def onnx_cell_spec_based_on_inputs(cell_idx_p, cell_idx_theta, onnx_folder):
    p_range = [-10, 10]
    p_num_bin = 128
    theta_range = [-30, 30]
    theta_num_bin = 128
    p_bins = np.linspace(p_range[0], p_range[1], p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    theta_bins = np.linspace(theta_range[0], theta_range[1], theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)


    p_lb, p_ub = p_lbs[cell_idx_p], p_lbs[cell_idx_p+1]
    theta_lb, theta_ub = theta_lbs[cell_idx_theta], theta_lbs[cell_idx_theta+1]

    # print(f"p_lb, p_ub: {p_lb, p_ub}")
    # print(f"theta_lb, theta_ub: {theta_lb, theta_ub}")

    # Open the .h5 file
    file_path = f'models/Linear/singlecell_{cell_idx_p}_{cell_idx_theta}.h5'

    # Reading the HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Read each dataset
        models_slope = file['modelslopes'][:]
        models_intercept = file['modelintercept'][()]
        ranges = file['ranges'][:]
        validp = file['validp'][:]
        validtheta = file['validtheta'][:]

    # print(f'models_slope: {models_slope}')
    # print(f'models_intercept: {models_intercept}')
    # print(f'ranges: {ranges}')
    # print(f'validp: {validp}')
    # print(f'validtheta: {validtheta}')


    # add skip connection to the NN model to pass the input to the output and then add linear layer to create linear constraints for the specification
    normalization_factor = np.array([6.36615, 17.247995])
    model_path = 'onnx/AllInOne.onnx'
    net = ONNXNet()
    onnx_model = onnx.load(model_path)
    weights = unpack_weights(onnx_model.graph.initializer)
    weights_keys = list(weights.keys())
    for o_weight, p_weight in zip(weights_keys[:-1], net.main.parameters()):
        p_weight.data = weights[o_weight]
    net.controller.weight.data = torch.FloatTensor([[-0.74, -0.44]])
    net.linear_weight.weight.data = torch.from_numpy((models_slope * normalization_factor).reshape(1, -1).astype(np.float32))
    net.linear_constraint.weight.data = torch.FloatTensor([[1.0, -1.0], [-1.0, 1.0]])

    onnx_file_name = f'{onnx_folder}/AllInOne_{cell_idx_p}_{cell_idx_theta}.onnx'
    x = torch.randn(1, 4)

    torch.onnx.export(net, 
                        x,
                        onnx_file_name,
                        input_names = ['input'],   # the model's input names
                        output_names = ['output']) # the model's output names
    

    input_bounds = np.array([[-0.8, 0.8], [-0.8, 0.8], [p_lb, p_ub]/normalization_factor[0], [theta_lb, theta_ub]/normalization_factor[1]], dtype=np.float32)
    # Diff = 'AllInOne'model's output – (A*states + b) -> Diff_max = ub, Diff_min = lb -> ub >= Y'_0 - (Y'_1 + b) , lb <= Y'_0 - (Y'_1 + b)
    # output constraint (safe condition, considering linear model: A*states + b + [lb, ub], and A*states = Y'_1, Y'= output of a layer before the last layer): Y'_1 + b + lb <= Y'_0 <= Y'_1 + b + ub
    # output constraint (unsafe condition): Y'_0 - ub >= Y'_1 + b or Y'_1 + b >= Y'_0 - lb -> Y'_0 - Y'_1 >= b + ub or -Y'_0 + Y'_1 >= -lb - b
    # continue: (unsafe condition, considering Y'_0 - Y'_1 = Y_0 or -Y'_0 + Y'_1 = Y_1):  Y_0 >= b + ub or Y_1 >= -lb - b
    output_bounds = np.array([ models_intercept + ranges[1], - ranges[0] - models_intercept], dtype=np.float32)
    return input_bounds, output_bounds, onnx_file_name


def vnnlib_cell_spec_based_on_io(cell_idx_p, cell_idx_theta, vnnlib_file_name, newhead=None):
    p_range = [-10, 10]
    p_num_bin = 128
    theta_range = [-30, 30]
    theta_num_bin = 128
    p_bins = np.linspace(p_range[0], p_range[1], p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    theta_bins = np.linspace(theta_range[0], theta_range[1], theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)


    p_lb, p_ub = p_lbs[cell_idx_p], p_lbs[cell_idx_p+1]
    theta_lb, theta_ub = theta_lbs[cell_idx_theta], theta_lbs[cell_idx_theta+1]

    # Open the .h5 file
    file_path = f'models/Linear/singlecell_{cell_idx_p}_{cell_idx_theta}.h5'

    # Reading the HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Read each dataset
        models_slope = file['modelslopes'][:]
        models_intercept = file['modelintercept'][()]
        ranges = file['ranges'][:]
        validp = file['validp'][:]
        validtheta = file['validtheta'][:]

    normalization_factor = np.array([6.36615, 17.247995])
    

    input_bounds = np.array([[-0.8, 0.8], [-0.8, 0.8], [p_lb, p_ub]/normalization_factor[0], [theta_lb, theta_ub]/normalization_factor[1]], dtype=np.float32)
    if newhead is not None:
        input_bounds = np.concatenate((newhead, input_bounds[2:, :]), axis=0)
    
    # GIVEN THE FOLLOWING: Diff = 'AllInOne'model's output – (A*states + b) | Diff_max = ub, Diff_min = lb, AllInOne'model's output = Y_0, states = [p, theta] = [X_2, X_3], A = models_slope, b = models_intercept 
    # safe condition: ub >= Y_0 - (A*states + b) , lb <= Y_0 - (A*states + b) ->  unsafe condition: ub <= Y_0 - (A*states + b) or lb >= Y_0 - (A*states + b)
    # rewrite the unsafe condition (OR of the following statements) as follows:
    # ub <= Y'_0 - (models_slope[0] * normalization_factor[0] * X_2 + models_slope[1] * normalization_factor[1] * X_1 + b) ,
    # lb >= Y'_0 - (models_slope[0] * normalization_factor[0] * X_0 + models_slope[1] * normalization_factor[1] * X_3 + b)


    with open(vnnlib_file_name, "w") as f:
        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        f.write(f"(declare-const Y_0 Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write(f"(assert (or")
        f.write(f"\n")
        f.write(f"    (and (<= ({ranges[1]} (- Y_0 (+ (* {models_slope[0]*normalization_factor[0]} X_2) (* {models_slope[1]*normalization_factor[1]} X_3) {models_intercept})))))\n")
        f.write(f"    (and (>= ({ranges[0]} (- Y_0 (+ (* {models_slope[0]*normalization_factor[0]} X_2) (* {models_slope[1]*normalization_factor[1]} X_3) {models_intercept})))))\n")
        f.write(f"))")
        f.write("\n")


if __name__ == '__main__':
              
    try:
        np.random.seed(seed=int(sys.argv[1]))
    except (IndexError, ValueError):
        raise ValueError("Expected seed (int) to be given as command line argument")
    

    onnx_folder = "onnx"
    files = os.listdir(onnx_folder)
    assert len(files) == 1, "Expected 1 onnx file in the onnx_files directory"

    # create a folder for the vnnlib files
    vnnlib_folder = "vnnlib"
    if not os.path.exists(vnnlib_folder):
        os.mkdir(vnnlib_folder)

    csv_path = "instances.csv"
    f = open(csv_path, "w")

    cell_list = [(10, 10), (30, 30), (30, 80), (50, 50), (50, 120), (80, 30), (80, 80), (120, 30), (120, 50), (120, 120)]
    timeout = 900
    newhead_dict = {}
    for cell_idx in cell_list:
        p_idx, theta_idx = cell_idx
        input_bounds, output_bounds, onnx_file_name = onnx_cell_spec_based_on_inputs(p_idx, theta_idx, onnx_folder)
        vnnlib_file_name = f"{vnnlib_folder}/prop_{p_idx}_{theta_idx}.vnnlib"
        save_vnnlib(input_bounds, output_bounds, vnnlib_file_name)
        f.write(f"{onnx_file_name},{vnnlib_file_name},{timeout}\n")

        num_instances = 5
        for idx in range(num_instances):
            negative_rnd = np.random.uniform(-0.8, -0.4, (2, 1))
            positive_rnd = np.random.uniform(0.4, 0.8, (2, 1))

            newhead = np.concatenate((negative_rnd, positive_rnd), axis=1)
            newhead_dict[(p_idx, theta_idx, idx)] = newhead

            input_bounds = np.concatenate((newhead, input_bounds[2:, :]), axis=0)
            vnnlib_file_name = f"{vnnlib_folder}/prop_{p_idx}_{theta_idx}_{idx}.vnnlib"
            save_vnnlib(input_bounds, output_bounds, vnnlib_file_name)

            f.write(f"{onnx_file_name},{vnnlib_file_name},{timeout}\n")
        
    f.close()


    # write the specification based on the input and output in a joint format without adding skip connetion to NN (for future support)
    csv_path = "instances_io.csv"
    f = open(csv_path, "w")

    cell_list = [(10, 10), (30, 30), (30, 80), (50, 50), (50, 120), (80, 30), (80, 80), (120, 30), (120, 50), (120, 120)]
    timeout = 900
    for cell_idx in cell_list:
        p_idx, theta_idx = cell_idx
        vnnlib_file_name = f"{vnnlib_folder}/prop_{p_idx}_{theta_idx}_io.vnnlib"
        vnnlib_cell_spec_based_on_io(p_idx, theta_idx, vnnlib_file_name)
        f.write(f"onnx/AllInOne.onnx,{vnnlib_file_name},{timeout}\n")

        num_instances = 5
        for idx in range(num_instances):
            vnnlib_file_name = f"{vnnlib_folder}/prop_{p_idx}_{theta_idx}_{idx}_io.vnnlib"
            newhead = newhead_dict[(p_idx, theta_idx, idx)]
            vnnlib_cell_spec_based_on_io(p_idx, theta_idx, vnnlib_file_name, newhead)
            f.write(f"onnx/AllInOne.onnx,{vnnlib_file_name},{timeout}\n")
        
    f.close()