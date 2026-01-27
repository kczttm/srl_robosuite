
import numpy as np
import time
import os
import yaml
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_and_confirm(data, save_path="figure.png"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # data shape: (maximum_step, 13)
    num_features = data.shape[1] 
    cols = 3 
    rows = math.ceil(num_features / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True) 
    axes = axes.flatten() # make indexing easy

    for i in range(num_features):
        axes[i].plot(data[:, i])
        axes[i].set_ylabel(f"Joint angle{i}")
        axes[i].grid(True)

    axes[-1].set_xlabel("Joint")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Ask user for confirmation
    answer = input("Does the figure look correct? (y/n): ").strip().lower()

    if answer != "y":
        print("Figure not approved. Exiting program.")
        sys.exit(1)

    print("Figure approved. Continuing execution.")

class Koopman_matrix(nn.Module):
    def __init__(self, state_dim, lifted_dim, is_idendity=False):
        super(Koopman_matrix, self).__init__()
        self.K_matrix = nn.Linear(state_dim + lifted_dim, state_dim + lifted_dim, bias=False)  # maybe the initial Koopman matrix should be an identity matrix?
    
        # Initialize as identity matrix
        if is_idendity:
            nn.init.eye_(self.K_matrix.weight)

    def forward(self, phi_x):
        return self.K_matrix(phi_x)
    
class NN_lifting(nn.Module):
    def __init__(self, state_dim, hidden_sizes, lifted_dim, nonlinearity='relu'):
        super(NN_lifting, self).__init__()
        self.state_dim = state_dim  # original state dim
        self.layer_sizes = (state_dim, ) + hidden_sizes + (lifted_dim,) # hidden layers
        
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])  # stack severeal layers together.
        
        # The weights are initialzied in default by:
#        stdv = 1. / math.sqrt(self.weight.size(1))
#        self.weight.data.uniform_(-stdv, stdv)
#        if self.bias is not None:
#           self.bias.data.uniform_(-stdv, stdv)

        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh   

    def forward(self, x):
        for i in range(len(self.fc_layers)-1):
            x = self.fc_layers[i](x)
            x = self.nonlinearity(x)
        out = self.fc_layers[-1](x)  # the last layer does not have an activation function, this may lift the NN outputs
        return out

# laod the pre-trained flow autoencoders
def load_Koopman_models(config, orig_dim, koopman_model_dir):
    lifting_func = NN_lifting(orig_dim, tuple(config['observable']['NN_model']['hidden_dim']), config['observable']['NN_model']['lifted_dim'], config['observable']['NN_model']['nonlinearity'])
    k_matrix = Koopman_matrix(orig_dim, config['observable']['NN_model']['lifted_dim'])

    device = 'cpu'

    # Load the weights
    trained_lifting_path = os.path.join(koopman_model_dir, 'Best_Lifting.pth')      # note that for the toilet task, we are using the Final model
    lifting_func.load_state_dict(torch.load(trained_lifting_path, weights_only=True))

    # Move to device if needed
    lifting_func = lifting_func.to(device)
    lifting_func.eval()  # Set to evaluation mode if you're not training

    trained_koopman_path = os.path.join(koopman_model_dir, 'Best_Koopman.pth')    
    k_matrix.load_state_dict(torch.load(trained_koopman_path, weights_only=True))

    # Move to device if needed
    k_matrix = k_matrix.to(device)
    k_matrix.eval()  # Set to evaluation mode if you're not training

    Koopman_model = {'lifting_func': lifting_func, 'k_matrix': k_matrix}

    return Koopman_model

def scale_to_match(flow_feature, action_norm, flow_norm):
    scale_factor = action_norm / (flow_norm + 1e-8)
    scaled_flow_feature = flow_feature * scale_factor
    return scaled_flow_feature

class KoopmanBiKinoPsyRightSideJointPosReaderDevice:
    """
    Device to read recorded joint positions (Right Arm + Right Hand) from an .npy file
    and serve them for playback.
    
    The .npy file is expected to have shape (T, 13) where:
    - Columns 0-6: Right Arm Joint Positions (7 DOF)
    - Columns 7-12: Right Hand Joint Positions (6 DOF) - [Index, Middle, Ring, Pinky, ThumbFlex, ThumbRot] (Assumption)
    """
    def __init__(self, initial_state, Koopman_model_dir, unscaled_initial_flow_feature, frequency=30.0, finger_rad = True, loop=False):
        """
        Args:
            npy_path (str): Path to the .npy file containing joint data.
            frequency (float): Expected playback frequency in Hz.
            loop (bool): Whether to loop the playback when finished.
        """
        self.initial_state_rad = initial_state.copy()
        self.initial_state = initial_state
        
        if finger_rad:
            self.initial_state[7:] = np.rad2deg(self.initial_state[7:])
        
        maximum_prediction = 150
        self.num_samples = maximum_prediction
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.loop = loop
        
        self.current_idx = 0
        self.start_time = None
        self.last_call_time = 0

        orig_dim = 141
        
        koopman_config_path = os.path.join(Koopman_model_dir, 'training_config.yaml')    
        with open(koopman_config_path, "r") as file:
            koopman_config = yaml.safe_load(file)
        
        self.koopman_models = load_Koopman_models(koopman_config, orig_dim, Koopman_model_dir)  # koopman_model = {'lifting_func': lifting_func, 'k_matrix': k_matrix}

        action_norm_path = os.path.join(Koopman_model_dir, 'action_norm.npy')    
        action_norm = np.load(action_norm_path)
        
        flow_norm_path = os.path.join(Koopman_model_dir, 'flow_norm.npy')    
        flow_norm = np.load(flow_norm_path)

        self.scaled_initial_flow_feature = scale_to_match(unscaled_initial_flow_feature, action_norm, flow_norm)

    def get_controller_state(self):
        """
        Returns the next joint configuration.
        
        Returns:
            dict or None: 
                {
                    'right_arm_positions': np.array(7),
                    'right_hand_positions': np.array(6)
                }
                Returns None if end of trajectory and not looping.
        """
        if self.start_time is None:
            self.start_time = time.time()
            return {
                'right_arm_positions': self.initial_state[:7],
                'right_hand_positions': self.initial_state[7:]
            }
        
        # Simple frame stepping
        if self.current_idx >= self.num_samples:
            if self.loop:
                self.current_idx = 0
                print("Looping playback...")
            else:
                return None
        
        sample = self.robot_action_pred[self.current_idx]
        self.current_idx += 1

        # Parse sample
        # Assuming layout: [Arm(7), Hand(6)]
        arm_positions = sample[:7]
        hand_positions = sample[7:]
        
        return {
            'right_arm_positions': arm_positions,
            'right_hand_positions': hand_positions
        }
    
    def reset(self):
        self.current_idx = 0
        self.start_time = None

    def begin_koopman_rollout(self, save_path):
        Robot_OriState = self.initial_state_rad

        init_original_x = torch.from_numpy(np.concatenate([Robot_OriState, self.scaled_initial_flow_feature])).float().to("cpu")
        init_lifted_x = self.koopman_models['lifting_func'](init_original_x)
        init_lifted_x = torch.cat((init_original_x, init_lifted_x))
        phi_x = init_lifted_x
        
        self.robot_action_pred = []
        for _ in range(self.num_samples):
            # Koopman rollout
            phi_x = self.koopman_models['k_matrix'](phi_x)
            phi_x_numpy = phi_x.cpu().detach().numpy()
            pred_robot_state = phi_x_numpy[:Robot_OriState.shape[0]]
            self.robot_action_pred.append(pred_robot_state)

        self.robot_action_pred = np.stack(self.robot_action_pred, axis=0)
        self.robot_action_pred[:, 7:] = np.rad2deg(self.robot_action_pred[:, 7:])

        visualize_and_confirm(self.robot_action_pred, os.path.join(save_path, "robot_action_prediction.png"))
