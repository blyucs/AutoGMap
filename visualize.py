import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as io

data_type = 'R882'
# data_type = 'R1484'
# data_type = 'R5828'

if data_type == 'R5828':
	matr = io.loadmat('R5828.mat')

	#假设字典中'data'是你要的数据：
	data = matr['R5828']
	#转换为.npy文件：
	A_plus = np.transpose(data.A)[0:22, 0:22]
	fixed_gap_size = 6
	# fixed_gap_size = 2
	data_size = A_plus.shape[0]
	# fill = 0
	fill = 1

elif data_type == 'R882':
	matr = io.loadmat('R882.mat')

	data = matr['R882']
	A_plus = np.transpose(data.A)

	fixed_gap_size = 6
	# fixed_gap_size = 2
	data_size = A_plus.shape[0]
	# fill = 0
	fill = 1
elif data_type == 'R1484':
	matr = io.loadmat('R1484.mat')

	data = matr['R882']
	A_plus = np.transpose(data.A)

	fixed_gap_size = 6
	# fixed_gap_size = 2
	data_size = A_plus.shape[0]
	# fill = 0
	fill = 1


#block-rl-20211021-200557/log.txt:28502:2021-10-21
# Action = [4, 4, 2, 2, 8, 2]  # *** 0.455
# Fill = [1, 0, 0, 1, 1]
# dyn_gap_count = 2
# fixed_gap_size = 6
# Coverage = 1.000
# Area = 0.455

#      2021-10-21 19:18:20,772
# Action = [8, 2, 2, 8, 2] # *** 0.471
# Fill = [0, 1, 1, 1]
# dyn_gap_count = 2
# fixed_gap_size = 6


# block-rl-20211021-145739
# Fill=[0, 1, 3, 3]
# Action=[8, 2, 2, 6, 4]
# dyn_gap_count = 4

# block-rl-20211022-103729
# Coverage:1.000, Area:0.430, Reward:0.914, Sparsity:0.692
# Fill = [1, 3, 0, 2, 3, 2]
# Action = [2, 2, 4, 2, 2, 6, 4]
# dyn_gap_count = 4



#2021-10-22 10:49:17,745  Decoder:[0, 0, 0, 1, 1, 1, 0, 0, 0, 1], # ***
# Fill = [0, 2, 2, 3]
# Action = [8, 2, 2, 8, 2] #, Coverage:1.000, Area:0.459, Reward:0.908, Sparsity:0.712
# dyn_gap_count = 4

#2021-10-22 13:33:39,424  Decoder:[0, 0, 0, 1, 1, 1, 0, 0, 1, 0],   # ***
# Fill = [1, 1, 3, 2]
# Action = [8, 2, 2, 6, 4] #, Coverage:1.000, Area:0.442, Reward:0.903, Sparsity:0.701
# dyn_gap_count = 4


# 2021-10-25 19:50:17,371  Decoder:[1, 1, 0, 1, 1, 1, 0, 0, 1, 0]
# Fill=[1, 3, 0, 2, 3, 2]
# Action=[2, 2, 4, 2, 2, 6, 4]
# Coverage=1.000
# Area=0.430
# Reward=0.893 # Sparsity:0.692


# 2021-10-25 20:19:40,075  Decoder:[1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
# Fill=[0, 2, 2]
# Action=[2, 2, 14, 4]
# Coverage=1.000
# Area=0.558
# Reward=0.888 # Sparsity:0.763


###############R882##################
# Fill=[3, 3, 2, 3, 2, 1, 1, 1, 2, 3, 3, 1, 1, 3, 2, 3, 1]
# Action=[32, 32, 64, 16, 48, 32, 16, 32, 96, 48, 32, 128, 16, 176, 16, 64, 16, 16]
# dyn_gap_count = 4

# Fill = [2, 2, 2, 3, 2, 2, 3, 2, 3]
# Action = [160, 96, 128, 128, 64, 96, 64, 32, 64, 32]

# Fill = [3, 2, 1, 3, 2, 2, 2, 2, 3, 2]
# Action = [32, 32, 160, 192, 96, 64, 64, 96, 64, 64, 18]

# Fill = [1, 2, 2, 2, 2]
# Action = [32, 224, 256, 64, 192, 114]

# Fill = [3, 2, 2, 3, 2, 3, 2]
# Action = [64, 256, 160, 96, 64, 96, 64, 82]



# 2021-10-24 12:17:19,238
Fill = [2, 3, 4, 2, 3, 4, 3]   # ****
Action = [32, 192, 160, 96, 160, 96, 64, 82]
Coverage = 1.000
Area = 0.225
Reward = 0.955 #, Sparsity:0.981



dyn_gap_count = 6
#2021-10-24 18:06:07,092
# Fill=[4, 5, 3, 3, 3, 2, 3, 4]
# Action=[32, 32, 160, 160, 160, 96, 160, 64, 18]
# Coverage=1.000
# Area=0.234
# Reward=0.953

# 2021-10-25 13:14:54,199  Decoder:[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1] (0.7 + 0.3)
# Fill=[4, 4, 2, 3, 3, 3, 2, 5, 3]    # ***
# Action=[32, 32, 160, 160, 128, 96, 96, 128, 32, 18]
# Coverage=0.995
# Area=0.200
# Reward=0.936 #Sparsity:0.979


# 2021-10-25 19:28:10,093  Decoder:[1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
# Fill=[2, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2] # ***
# Action=[32, 32, 32, 192, 96, 96, 64, 64, 96, 96, 64, 18]
# Coverage=0.998
# Area=0.196
# Reward=0.940 # Sparsity:0.978


# 2021-10-25 18:24:16,696  Decoder:[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
# Fill=[2, 2, 2, 2, 2, 2, 2, 2, 3, 2]   # ***
# Action=[32, 128, 96, 128, 96, 64, 64, 96, 96, 32, 50]
# Coverage=0.998
# Area=0.204
# Reward=0.958 #Sparsity:0.979

# 2021-10-25 13:14:54,199  Decoder:[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
# Fill=[4, 4, 2, 3, 3, 3, 2, 5, 3]
# Action=[32, 32, 160, 160, 128, 96, 96, 128, 32, 18]
# Coverage=0.995
# Area=0.200
# Reward=0.936 #Sparsity:0.979


###############R1484##################
# dyn_gap_count = 4
# dyn_gap_count = 6


#2021-10-27 16:05:21,850  Decoder:[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0]
# Fill=[1, 2, 1, 2, 2, 3, 3, 2, 2, 3, 2, 3, 1, 2, 3, 2, 2, 3]
# Action=[96, 32, 32, 288, 192, 160, 64, 32, 64, 64, 32, 64, 32, 128, 32, 32, 64, 32, 44]
# Coverage=0.992
# Area=0.148
# Reward=0.950 # Sparsity:0.981


# 2021-10-27 18:09:22,826  Decoder:[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1]
# Fill=[2, 2, 2, 2, 3, 2, 3, 2, 3, 1, 3, 3, 2, 2]
# Action=[96, 64, 288, 192, 128, 96, 128, 32, 96, 32, 128, 64, 32, 96, 12]
# Coverage=0.999
# Area=0.185
# Reward=0.963 #Sparsity:0.985



# 2021-10-26 23:19:45,307  Decoder:[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]
# Fill=[5, 2, 2, 3, 3, 2, 2, 4, 0, 2, 4]
# Action=[128, 224, 288, 224, 160, 64, 64, 160, 64, 32, 64, 12]
# Coverage=0.993
# Area=0.173
# Reward=0.944 #Sparsity:0.984


# 2021-10-26 10:53:41,497  Decoder:[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]
# Fill=[4, 5, 2, 4, 4, 4, 5, 3, 2, 2, 3, 3, 3]
# Action=[32, 96, 256, 288, 128, 96, 64, 64, 128, 160, 64, 32, 64, 12]
# Coverage=1.000
# Area=0.171
# Reward=0.966 # Sparsity:0.984

if dyn_gap_count == 4:
	# gap_case = [0, 0.25, 0.5, 1.0]
	gap_case = [0, 0.25, 0.75, 1.0]
elif dyn_gap_count == 6:
	gap_case = [0, 1/5, 2/5, 3/5, 4/5, 5/5]

def visualize(action, fill_action, Coverage, Area, dyn_gap_count, fixed_gap_size):
	# plt.spy(A_plus)
	if data_type == 'R882':
		plt.figure(dpi=400)
		plt.spy(A_plus, markersize=0.5)
	elif data_type == 'R1484':
		plt.figure(dpi=300)
		plt.spy(A_plus, markersize=0.5)
	else:
		plt.figure()
		plt.spy(A_plus)
	start = 0
	for i in range(len(action)):
		end = start+action[i]-1
		x = [start-0.5, end+0.5, end+0.5, start-0.5] # clock-wise  [a,b]
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

					x = [left - 0.5, start - 0.5, start - 0.5, left - 0.5]  # clock-wise  [a,b]
					y = [start - 0.5, start - 0.5, down + 0.5, down + 0.5]

					plt.fill(x, y, alpha=0.9)

				if fill_action[i-1] != 0:
					if dyn_gap_count == 2:  # only 0/1 , fill or not
						new_size = fixed_gap_size if fixed_gap_size < action[i] else action[i]
					else:  # fill with 0;1/4;1/2;1
						new_size = math.floor(gap_case[fill_action[i-1]] * action[i])

					up = 0 if start - new_size < 0 else start - new_size
					right = start + (start - up) - 1

					x = [start - 0.5, right + 0.5, right + 0.5, start - 0.5]  # clock-wise  [a,b]
					y = [up - 0.5, up - 0.5, start - 0.5, start - 0.5]

					plt.fill(x, y, alpha=0.6)

		start = end +1  # for [a,b]
	plt.text(600,100,"Coverage:{}".format(Coverage))
	plt.text(600,150,"Area:{}".format(Area))
	# plt.text(600,200,"Reward:{}".format(Reward))
	plt.show()

if __name__ == '__main__':
	visualize(Action, Fill, Coverage, Area, dyn_gap_count, fixed_gap_size)