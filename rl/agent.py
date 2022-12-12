"""Reinforcement Learning-based agent"""

from .gradient_estimators import REINFORCE, PPO


def create_agent(
        op_size,
        hidden_size,
        num_lstm_layers,
        lr_ctrl,
        bl_dec,
        agent_ctrl,
        action_len,
        fill,
        dyn_gap_count,
        model_type
        ):
    """Create Agent

    Args:
      op_size (int) : number of unique operations
      hidden_size (int) : number of neurons in RNN's hidden layer
      num_lstm_layers (int) : number of LSTM layers
      num_cells (int) : number of decoder layers
      num_branches (int) : number of layers in a cell
      lr_ctrl (float) : controller's learning rate
      bl_dec (float) : baseline's decay
      agent_ctrl (str) : type of agent's controller
      num_enc_nodes (int) : size of initial sampling pool

    Returns:
      controller net that provides the sample() method
      gradient estimator

    """
    if model_type == 'LSTM':
        from rl.micro_controllers import MicroController as Controller
    else:
        from rl.micro_controllers import MicroControllerCell as Controller

    controller = Controller(
        # num_enc_nodes,
        op_size,
        hidden_size,
        num_lstm_layers, action_len, fill, dyn_gap_count)
        # num_dec_layers=num_cells,
        # num_ctx_layers=num_branches)
    if agent_ctrl == 'ppo':
        agent = PPO(
            controller,
            clip_param=0.1,
            lr=lr_ctrl,
            baseline_decay=bl_dec,
            action_size=controller.action_size())
    elif agent_ctrl == 'reinforce':
        agent = REINFORCE(
            controller,
            lr=lr_ctrl,
            baseline_decay=bl_dec)
    return agent


def train_agent(agent, sample):
    """Training controller"""
    config, left_config, reward, entropy, log_prob = sample
    action = config
    loss, dist_entropy = agent.update((reward, action, left_config, entropy, log_prob))
