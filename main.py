import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import os
import argparse
from numpy import array
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import time
from rl.agent import create_agent, train_agent
import math
import utils
import sys
import glob
from tensorboardX import SummaryWriter
import matplotlib.animation as animation

# CONTROLLER
BL_DEC = 0.95
# AGENT_CTRL = 'ppo'
AGENT_CTRL = 'reinforce'
LR_CTRL = 1e-4
os.environ["CUDA_VISIBLE_DEVICES"]="3"
logging.basicConfig(level=logging.INFO)
OP_SIZE = 2
def get_arguments():
	"""Parse all the arguments provided from the CLI.

	Returns:
	  A list of parsed arguments.
	"""
	parser = argparse.ArgumentParser(description="NAS Search")

	# Controller
	parser.add_argument("--hidden_size", type=int, default=10,
	                    help="Number of neurons in the controller's RNN.")
	parser.add_argument("--num_lstm_layers", type=int, default=1,
	                    help="Number of layers in the controller.")
	parser.add_argument("--bl-dec", type=float, default=BL_DEC,
	                    help="Baseline decay.")
	parser.add_argument("--agent-ctrl", type=str, default=AGENT_CTRL,
	                    help="Gradient estimator algorithm")
	parser.add_argument("--lr-ctrl", type=float, default=LR_CTRL,
                    help="Learning rate for controller.")
	parser.add_argument("--op-size", type=int, default=OP_SIZE,
	                    help="Number of unique operations.")
	parser.add_argument('--save', type=str, default='EXP/', help='experiment name')
	return parser.parse_args()


args = get_arguments()
args.save = '{}block-rl-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('./rl/*.py',recursive=True)+glob.glob('*.py',recursive=True))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

import scipy.io as io
data_type = 'R5828'

if data_type == 'R5828':
	matr = io.loadmat('R5828.mat')
	data = matr['R5828']
	A_plus = np.transpose(data.A)[0:22, 0:22]

	fixed_gap_size = 6
	grid_size = 2
	data_size = A_plus.shape[0]
	action_len = math.ceil(data_size/grid_size) - 1
	fill = 1
	# dyn_gap_count = 4
	# dyn_gap_count = 6
	dyn_gap_count = 2
	model_type = 'LSTM'
	# model_type = 'BiLSTM'

	if dyn_gap_count == 4:
		# gap_case = [0, 0.25, 0.5, 1.0]
		gap_case = [0, 0.25, 0.75, 1.0]
	elif dyn_gap_count == 6:
		gap_case = [0, 1/5, 2/5, 3/5, 4/5, 5/5]
	EPOCHS = 40000
elif data_type == 'R882':
	matr= io.loadmat('R882.mat')
	data = matr['R882']
	A_plus = np.transpose(data.A)

	fixed_gap_size = 6
	# fixed_gap_size = 2
	grid_size = 32
	# grid_size = 16
	data_size = A_plus.shape[0]
	action_len = math.ceil(data_size/grid_size)-1
	# fill = 0
	fill = 1
	# dyn_gap_count = 6
	dyn_gap_count = 4
	# dyn_gap_count = 2
	model_type = 'LSTM'
	# model_type = 'BiLSTM'

	if dyn_gap_count == 4:
		# gap_case = [0, 0.25, 0.5, 1.0]
		gap_case = [0, 0.25, 0.75, 1.0]
	elif dyn_gap_count == 6:
		gap_case = [0, 1/5, 2/5, 3/5, 4/5, 5/5]

	EPOCHS = 50000
elif data_type == 'R1484':
	matr= io.loadmat('R1484.mat')
	data = matr['R882']
	A_plus = np.transpose(data.A)

	fixed_gap_size = 6
	# fixed_gap_size = 2
	grid_size = 32
	# grid_size = 16
	data_size = A_plus.shape[0]
	action_len = math.ceil(data_size/grid_size)-1
	# fill = 0
	fill = 1
	dyn_gap_count = 6
	# dyn_gap_count = 4
	# dyn_gap_count = 2
	model_type = 'LSTM'
	# model_type = 'BiLSTM'
	EPOCHS = 50000

if dyn_gap_count == 4:
	# gap_case = [0, 0.25, 0.5, 1.0]
	gap_case = [0, 0.25, 0.75, 1.0]
elif dyn_gap_count == 6:
	gap_case = [0, 1/5, 2/5, 3/5, 4/5, 5/5]


Coverage_writer = SummaryWriter(logdir='{}/tb/Coverage'.format(args.save))
Area_writer = SummaryWriter(logdir='{}/tb/Area'.format(args.save))
Reward_writer = SummaryWriter(logdir='{}/tb/Reward'.format(args.save))
visual = []
ani_interval = 1000
def main():
	exp_name = time.strftime('%H_%M_%S')
	logging.info(" Running Experiment {}".format(exp_name))

	total_nozero_count = np.count_nonzero(A_plus)
	total_area = math.pow(data_size, 2)
	# generate controller / rl-agent
	agent = create_agent(
		args.op_size,
		args.hidden_size,
		args.num_lstm_layers,
		args.lr_ctrl,
		args.bl_dec,
		args.agent_ctrl,
		action_len,
		fill,
		dyn_gap_count,
		model_type
	)
	logging.info(" Training Controller")
	for epoch in range(0, EPOCHS):
		# sample
		decoder_config, fill_config, entropy, log_prob = agent.controller.sample()
		# evaluate
		# config to action
		action = []
		fill_action = []
		tmp = grid_size
		for i in range(action_len):
			add_size = grid_size if i != action_len - 1 else data_size - action_len * grid_size
			if decoder_config[i] == 1:
				action.append(tmp)
				tmp = add_size

				if fill:
					fill_action.append(fill_config[i])
			else:
				tmp+= add_size
		action.append(tmp)

		# action to reward
		start = 0
		A_blk_count = 0
		A_blk_area = 0
		for i in range(len(action)):
			end = start+action[i]
			A_blk_count += np.count_nonzero(A_plus[start:end, start:end])
			A_blk_area += math.pow(action[i],2)
			if fill:
				if i != 0:  # do not care the first one
					if fill_action[i - 1] != 0:
						if dyn_gap_count == 2:  # only 0/1 , fill or not
							new_size = fixed_gap_size if fixed_gap_size < action[i] else action[i]
						else:  # fill with 0;1/4;1/2;1
							new_size = math.floor(gap_case[fill_action[i-1]] * action[i])
						# left
						left = 0 if start - new_size < 0 else start - new_size
						down = start + (start - left)

						A_blk_count += np.count_nonzero(A_plus[start:down, left:start])
						A_blk_area += math.pow( start - left, 2) # bugfix

					# right
					if fill_action[i-1] != 0:
						if dyn_gap_count == 2:  # only 0/1 , fill or not
							new_size = fixed_gap_size if fixed_gap_size < action[i] else action[i]
						else:  # fill with 0;1/4;1/2;1
							new_size = math.floor(gap_case[fill_action[i-1]] * action[i])
						up = 0 if start - new_size < 0 else start - new_size
						right = start + (start - up)

						A_blk_count += np.count_nonzero(A_plus[up:start, start:right])
						A_blk_area += math.pow( start - up, 2)

			start = end

		reward1 = A_blk_count/total_nozero_count
		reward2 = 1 - A_blk_area/total_area
		Area_ratio = A_blk_area/total_area
		sparsity = 1-A_blk_count / A_blk_area
		reward = 0.8*reward1 + 0.2*reward2
		# train controller
		sample = (decoder_config, fill_config, reward, entropy, log_prob)
		train_agent(agent, sample)
		logging.info(' Decoder:%s Fill=%s Action=%s Coverage=%.3f Area=%.3f Reward=%.3f Sparsity:%.3f',
		             decoder_config, fill_action, action, reward1, Area_ratio, reward, sparsity)

		Coverage_writer.add_scalar('block_schedule_rl_training', reward1, epoch)
		Area_writer.add_scalar('block_schedule_rl_training', Area_ratio, epoch)
		Reward_writer.add_scalar('block_schedule_rl_training', reward, epoch)

		if epoch < 20000:
			ani_interval = 1000
		elif epoch < 30000:
			ani_interval = 500
		elif epoch < 40000:
			ani_interval = 300
		elif epoch < 45000:
			ani_interval = 200
		elif epoch < 49000:
			ani_interval = 100
		else:
			ani_interval = 50

		if epoch % ani_interval == 0:
			visual.append((epoch, action, fill_action, reward1, Area_ratio, reward, dyn_gap_count, fixed_gap_size))

ax = plt.figure(dpi=400)
def visualize(n):
	plt.clf()
	plt.spy(A_plus, markersize=0.5)
	epoch, action, fill_action, Coverage, Area, Reward, dyn_gap_count, fixed_gap_size = visual[n]
	start = 0
	for i in range(len(action)):
		end = start+action[i]-1
		x = [start-0.5, end+0.5, end+0.5, start-0.5]
		y = [start-0.5, start-0.5, end+0.5, end+0.5]

		plt.fill(x,y, alpha = 0.9)
		if fill:
			if i != 0:  # do not care the first one
				if fill_action[i -1] != 0:
					if dyn_gap_count == 2:  # only 0/1 , fill or not
						new_size = fixed_gap_size if fixed_gap_size < action[i] else action[i]
					else:  # fill with 0;1/4;1/2;1
						new_size = math.floor(gap_case[fill_action[i-1]] * action[i])
					left = 0 if start - new_size < 0 else start - new_size
					down = start + (start - left) - 1

					x = [left - 0.5, start - 0.5, start - 0.5, left - 0.5]
					y = [start - 0.5, start - 0.5, down + 0.5, down + 0.5]

					plt.fill(x, y, alpha=0.9)

				if fill_action[i-1] != 0:
					if dyn_gap_count == 2:  # only 0/1 , fill or not
						new_size = fixed_gap_size if fixed_gap_size < action[i] else action[i]
					else:  # fill with 0;1/4;1/2;1
						new_size = math.floor(gap_case[fill_action[i-1]] * action[i])
					up = 0 if start - new_size < 0 else start - new_size
					right = start + (start - up) - 1

					x = [start - 0.5, right + 0.5, right + 0.5, start - 0.5]
					y = [up - 0.5, up - 0.5, start - 0.5, start - 0.5]

					plt.fill(x, y, alpha=0.9)

		start = end +1
	plt.text(600,100,"Epoch:   {}".format(epoch))
	plt.text(600,150,"Coverage:{:.3f}".format(Coverage))
	plt.text(600,200,"Area:    {:.3f}".format(Area))
	plt.text(600,250,"Reward:  {:.3f}".format(Reward))

	# plt.show()

if __name__ == '__main__':
	main()
	ani = animation.FuncAnimation(ax, visualize, frames=len(visual), interval=400)
	ani.save('MovWave.gif')
	# plt.show()