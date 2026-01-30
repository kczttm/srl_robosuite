
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

def compare_fk_trajectory(sim, pred_qpos_traj, body_name, save_path="figure.png", gt_qpos_traj = None):
    """
    Compute EE pose trajectory for a sequence of joint angles.

    sim         : robosuite sim object
    qpos_traj   : numpy array of shape (T, nq)
    body_name   : name of the body for FK (e.g., 'gripper0_right_eef')
    """

    # Save original simulation state 
    original_state = sim.get_state()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # data shape: (maximum_step, 13)
    num_features = 7
    robot_state_name = (
        [f'pos {i}' for i in range(1, 4)] + [f'quat {i}' for i in range(1, 5)]
    )

    cols = 3
    rows = math.ceil(num_features / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2.5*rows), sharex=True) 
    axes = axes.flatten() # make indexing easy

    T = pred_qpos_traj.shape[0]

    pos_traj = np.zeros((T, 3))
    rot_traj = np.zeros((T, 3, 3))
    quat_traj = np.zeros((T, 4))

    body_id = sim.model.body_name2id(body_name)

    for t in range(T):
        # Set joint configuration
        sim.data.qpos[:7] = pred_qpos_traj[t,:7]
        sim.forward()

        # Extract pose
        pos_traj[t] = sim.data.body_xpos[body_id].copy()
        rot_traj[t] = sim.data.body_xmat[body_id].reshape(3, 3).copy()
        quat_traj[t] = sim.data.body_xquat[body_id].copy()

    plot_traj = np.concatenate([pos_traj, quat_traj], axis=1)

    for i in range(num_features):
        axes[i].plot(
            plot_traj[:, i],  # finger joints in deg
            color="tab:blue",
            lw=1.8,
            label="Pred" if i == 0 else None,
        )

        axes[i].set_ylabel(f"value {i}", fontsize=9)
        axes[i].set_title(robot_state_name[i], fontsize=8)
        axes[i].grid(True, alpha=0.3)

    if gt_qpos_traj is not None:
        T = gt_qpos_traj.shape[0]

        pos_traj = np.zeros((T, 3))
        rot_traj = np.zeros((T, 3, 3))
        quat_traj = np.zeros((T, 4))

        body_id = sim.model.body_name2id(body_name)

        for t in range(T):
            # Set joint configuration
            sim.data.qpos[:7] = gt_qpos_traj[t,:7]
            sim.forward()

            # Extract pose
            pos_traj[t] = sim.data.body_xpos[body_id].copy()
            rot_traj[t] = sim.data.body_xmat[body_id].reshape(3, 3).copy()
            quat_traj[t] = sim.data.body_xquat[body_id].copy()

        plot_traj = np.concatenate([pos_traj, quat_traj], axis=1)

        for i in range(num_features):
            axes[i].plot(
                plot_traj[:, i],  # finger joints in deg
                color="tab:red",
                lw=1.5,
                alpha=0.7,
                linestyle="--",
                label="GT" if i == 0 else None,
            )

            axes[i].set_ylabel(f"value {i}", fontsize=9)
            axes[i].set_title(robot_state_name[i], fontsize=8)
            axes[i].grid(True, alpha=0.3)        

    # show legend only once
    axes[0].legend(fontsize=8, loc="upper right")

    # Hide unused axes
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Restore original simulation state
    sim.set_state(original_state) 
    sim.forward()

def visualize_and_confirm(data, save_path="figure.png", compare_GT_actions=None):
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # data shape: (maximum_step, 13)
    num_features = data.shape[1] 
    robot_state_name = (
        [f'robot state {i}' for i in range(1, num_features + 1)]
    )
    cols = 5 
    rows = math.ceil(num_features / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2.5*rows), sharex=True) 
    axes = axes.flatten() # make indexing easy

    for i in range(num_features):
        axes[i].plot(
            data[:, i],  # finger joints in deg
            color="tab:blue",
            lw=1.8,
            label="Pred" if i == 0 else None,
        )

        if compare_GT_actions is not None:
            axes[i].plot(
                compare_GT_actions[:, i],
                color="tab:red",
                lw=1.5,
                alpha=0.7,
                linestyle="--",
                label="GT" if i == 0 else None,
            )

        axes[i].set_ylabel(f"Joint {i}", fontsize=9)
        axes[i].set_title(robot_state_name[i], fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Joint")

    # show legend only once
    axes[0].legend(fontsize=8, loc="upper right")

    # Hide unused axes
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

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
    def __init__(self, initial_state, Koopman_model_dir, unscaled_initial_flow_feature, compare_GT_actions=None, frequency=30.0, finger_rad = True, loop=False):
        """
        Args:
            npy_path (str): Path to the .npy file containing joint data.
            frequency (float): Expected playback frequency in Hz.
            loop (bool): Whether to loop the playback when finished.
        """
        self.initial_state_rad = initial_state.copy()
        self.initial_state = initial_state
        self.compare_GT_actions = compare_GT_actions

        if finger_rad:
            self.initial_state[7:] = np.rad2deg(self.initial_state[7:])
            if compare_GT_actions is not None:
                self.compare_GT_actions[:, 7:] = np.rad2deg(self.compare_GT_actions[:, 7:])

        # maximum_prediction = 160 # for cloth uncovering
        maximum_prediction = 160 # for box opening
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

    def begin_koopman_rollout(self, save_path, env_sim, ee_name):
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
        
        compare_fk_trajectory(env_sim, self.robot_action_pred, ee_name, os.path.join(save_path, "end_effector_position.png"), self.compare_GT_actions)
        visualize_and_confirm(self.robot_action_pred, os.path.join(save_path, "robot_action_prediction.png"), self.compare_GT_actions)
