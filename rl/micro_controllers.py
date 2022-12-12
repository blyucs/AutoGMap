"""RL Controller"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_long(x):
    x = torch.tensor(x, dtype=torch.long)
    return x.view((1, 1))


class MicroController(nn.Module):
    """ Stack LSTM controller based on ENAS

    https://arxiv.org/abs/1802.03268

    With modification that indices and ops chosen with linear classifier
    Samples output strides for encoder and decoder structure
    """
    def __init__(self, op_size,
                 hidden_size=100, num_lstm_layers=2, action_len = 0, fill = 1, dyn_gap_count = 0
                ):
        """
        Args:
          num_enc_scales (int): encoder input scales
          op_size (int): numebr of operations
          hidden_size (int): number of hidden units of LSTM
          num_lstm_layers (int): number of LSTM layers
          num_dec_layers (int): number of cells in the decoder
          num_ctx_layers (int): numebr of layers in each cell
        """
        super(MicroController, self).__init__()

        # additional configs
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        self.temperature = None
        self.tanh_constant = None

        # the network
        self.rnn = nn.LSTM(hidden_size, hidden_size,
                           num_lstm_layers, bidirectional=False)  # apply the nn.LSTM as nn.LSTMCell
        self.direction_num = 1
        # self.enc_op = nn.Embedding(op_size, hidden_size)
        self.g_emb = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self._action_len = action_len # action size not determine
        self.fill = fill
        self.dyn_gap_count = dyn_gap_count

        # contextual predictions
        ctx_fcs = []
        for _ in range(self._action_len):
                ctx_fcs.append(nn.Linear(hidden_size, op_size)) # for 2 = 2, 3 = 5, 4 = 8, etc.
        self.ctx_fcs = nn.ModuleList(ctx_fcs)
        # left fcs
        left_fcs = []
        for _ in range(self._action_len):
            left_fcs.append(nn.Linear(hidden_size, self.dyn_gap_count)) # for 2 = 2, 3 = 5, 4 = 8, etc.
        self.left_fcs = nn.ModuleList(left_fcs)
        # right fcs
        right_fcs = []
        for _ in range(self._action_len):
            right_fcs.append(nn.Linear(hidden_size, self.dyn_gap_count)) # for 2 = 2, 3 = 5, 4 = 8, etc.
        self.right_fcs = nn.ModuleList(right_fcs)

        # init parameters
        self.reset_parameters()

    def action_size(self):
        return self._action_len

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def sample(self):
        """Sample one architecture
        """
        return self.forward()

    def evaluate(self, config, left_config):
        """Evaluate entropy entropy and log probability of given architecture."""
        return self.forward(config, left_config)

    def forward(self, config=None, l_config = None):
        """ sample a decoder or compute the log_prob if decoder config is given
        Args:
          config (List): decoder architecture

        Returns:
          dec_arch: (ctx, conns)
            ctx: (index, op) x 4
            conns: [index_1, index_2] x 3
          entropy
          log_prob
        """
        do_sample = config is None
        if do_sample:
            ctx_config = []
            left_config = []
        else:
            dec_arch = config
            ctx_config = dec_arch
            left_config = l_config
        inputs = self.g_emb

        hidden = (torch.zeros([self.num_lstm_layers * self.direction_num, 1, self.hidden_size]),
                  torch.zeros([self.num_lstm_layers * self.direction_num, 1, self.hidden_size]))
        # hidden = torch.zeros([self.num_lstm_layers, 1, self.hidden_size])

        entropy = 0
        log_prob = 0

        def calc_prob(critic_logits, x):
            """compute entropy and log_prob."""
            softmax_logits, log_softmax_logits = critic_logits
            ent = softmax_logits * log_softmax_logits
            ent = -1 * ent.sum()
            log_prob = -F.nll_loss(log_softmax_logits, x.view(1))
            return ent, log_prob

        def compute_critic_logits(logits):
            softmax_logits = F.softmax(logits, dim=-1)
            log_softmax_logits = F.log_softmax(logits, dim=-1)
            critic_logits = (softmax_logits, log_softmax_logits)
            return critic_logits

        def sample_logits(critic_logits):
            softmax_logits = critic_logits[0]
            x = softmax_logits.multinomial(num_samples=1)
            ent, log_prob = calc_prob(critic_logits, x)
            return x, ent, log_prob

        # sample contextual cell
        for i in range(self._action_len):
            output, hidden = self.rnn(inputs, hidden)
            logits = self.ctx_fcs[i](output.squeeze(0))
            critic_logits = (compute_critic_logits(logits))
            if do_sample:
                pos, curr_ent, curr_log_prob = sample_logits(critic_logits)
                ctx_config.append(int(pos))
            else:
                pos = torch_long(ctx_config[i])
                curr_ent, curr_log_prob = calc_prob(critic_logits, pos)
            entropy += curr_ent
            log_prob += curr_log_prob
            inputs = output

            if self.fill:
                if int(pos) == 1:
                    #left
                    output, hidden = self.rnn(inputs, hidden)
                    logits = self.left_fcs[i](output.squeeze(0))
                    critic_logits = (compute_critic_logits(logits))
                    if do_sample:
                        pos, curr_ent, curr_log_prob = sample_logits(critic_logits)
                        left_config.append(int(pos))
                    else:
                        pos = torch_long(left_config[i])
                        curr_ent, curr_log_prob = calc_prob(critic_logits, pos)
                    entropy += curr_ent
                    log_prob += curr_log_prob
                    inputs = output
                else:
                    if do_sample:
                        left_config.append(int(0))
        return ctx_config, left_config, entropy, log_prob

    def evaluate_actions(self, actions_batch):
        log_probs, entropies = [], []
        action_length, action_size = actions_batch.shape
        for i in range(action_length):
            _, entropy, log_prob = self.evaluate(actions_batch[i])
            log_probs.append(log_prob.view(1))
            entropies.append(entropy.view(1))
        return torch.cat(log_probs), torch.cat(entropies)


class MicroControllerCell(nn.Module):
    """ Stack LSTM controller based on ENAS

    https://arxiv.org/abs/1802.03268

    With modification that indices and ops chosen with linear classifier
    Samples output strides for encoder and decoder structure
    """
    def __init__(self, op_size,
                 hidden_size=100, num_lstm_layers=2, action_len = 0, fill = 1, dyn_gap_count = 0
                 ):
        """
        Args:
          num_enc_scales (int): encoder input scales
          op_size (int): numebr of operations
          hidden_size (int): number of hidden units of LSTM
          num_lstm_layers (int): number of LSTM layers
          num_dec_layers (int): number of cells in the decoder
          num_ctx_layers (int): numebr of layers in each cell
        """
        super(MicroControllerCell, self).__init__()

        # additional configs
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        self.temperature = None
        self.tanh_constant = None

        # the network
        self.rnn = nn.LSTMCell(hidden_size, hidden_size
                           )  # apply the nn.LSTM as nn.LSTMCell
        self.g_emb = nn.Parameter(torch.zeros(1, hidden_size))
        self._action_len = action_len # action size not determine
        self.fill = fill
        self.dyn_gap_count = dyn_gap_count
        self.direction_num = 2

        # contextual predictions
        ctx_fcs = []
        for _ in range(self._action_len):
            ctx_fcs.append(nn.Linear(hidden_size * self.direction_num, op_size))

        # left fcs
        left_fcs = []
        for _ in range(self._action_len):
            left_fcs.append(nn.Linear(hidden_size * self.direction_num, self.dyn_gap_count))
        self.left_fcs = nn.ModuleList(left_fcs)

        # init parameters
        self.reset_parameters()

    def action_size(self):
        return self._action_len

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def sample(self):
        """Sample one architecture
        """
        return self.forward()

    def evaluate(self, config, left_config):
        """Evaluate entropy entropy and log probability of given architecture."""
        # config = MicroController.action2config(
        #     action, dec_block=self.num_dec_layers, ctx_block=self.num_ctx_layers)  # comments by l
        return self.forward(config, left_config)

    def forward(self, config=None, l_config = None):
        """ sample a decoder or compute the log_prob if decoder config is given
        Args:
          config (List): decoder architecture

        Returns:
          dec_arch: (ctx, conns)
            ctx: (index, op) x 4
            conns: [index_1, index_2] x 3
          entropy
          log_prob
        """
        do_sample = config is None
        if do_sample:
            ctx_config = []
            left_config = []
        else:
            dec_arch = config
            ctx_config = dec_arch
            left_config = l_config
        inputs = self.g_emb

        hidden = (torch.zeros([ 1, self.hidden_size]),
                  torch.zeros([ 1, self.hidden_size]))

        entropy = 0
        log_prob = 0

        def calc_prob(critic_logits, x):
            """compute entropy and log_prob."""
            softmax_logits, log_softmax_logits = critic_logits
            ent = softmax_logits * log_softmax_logits
            ent = -1 * ent.sum()
            log_prob = -F.nll_loss(log_softmax_logits, x.view(1))
            return ent, log_prob

        def compute_critic_logits(logits):
            softmax_logits = F.softmax(logits, dim=-1)
            log_softmax_logits = F.log_softmax(logits, dim=-1)
            critic_logits = (softmax_logits, log_softmax_logits)
            return critic_logits

        def sample_logits(critic_logits):
            softmax_logits = critic_logits[0]
            x = softmax_logits.multinomial(num_samples=1)
            ent, log_prob = calc_prob(critic_logits, x)
            return x, ent, log_prob

        forward_output = []
        backward_output = []
        forward_fill_output = []
        backward_fill_output = []
        for i in range(self._action_len):
            hx, cx = self.rnn(inputs, hidden) # without output, output need to collect by yourself
            forward_output.append(hx)
            inputs = hx
            hidden = (hx, cx)

            if self.fill:
                hx, cx = self.rnn(inputs, hidden) # without output, output need to collect by yourself
                forward_fill_output.append(hx)
                inputs = hx
                hidden = (hx, cx)


        for i in range(self._action_len):
            hx, cx = self.rnn(inputs, hidden) # without output, output need to collect by yourself
            backward_output.append(hx)
            inputs = hx
            hidden = (hx, cx)
            if self.fill:
                hx, cx = self.rnn(inputs, hidden) # without output, output need to collect by yourself
                backward_fill_output.append(hx)
                inputs = hx
                hidden = (hx, cx)

        # sample contextual cell
        for i in range(self._action_len):
            logits = self.ctx_fcs[i]( torch.cat([forward_output[i],backward_output[self._action_len - 1 -i]],1) )
            critic_logits = (compute_critic_logits(logits))
            if do_sample:
                pos, curr_ent, curr_log_prob = sample_logits(critic_logits)
                ctx_config.append(int(pos))
            else:
                pos = torch_long(ctx_config[i])
                curr_ent, curr_log_prob = calc_prob(critic_logits, pos)
            entropy += curr_ent
            log_prob += curr_log_prob

            if self.fill:
                if int(pos) == 1:
                    logits = self.left_fcs[i]( torch.cat([forward_fill_output[i],backward_fill_output[self._action_len - 1 -i]],1))
                    critic_logits = (compute_critic_logits(logits))
                    if do_sample:
                        pos, curr_ent, curr_log_prob = sample_logits(critic_logits)
                        left_config.append(int(pos))
                    else:
                        pos = torch_long(left_config[i])
                        curr_ent, curr_log_prob = calc_prob(critic_logits, pos)
                    entropy += curr_ent
                    log_prob += curr_log_prob
                else:
                    if do_sample:
                        left_config.append(int(0))
        return ctx_config, left_config, entropy, log_prob

    def evaluate_actions(self, actions_batch):
        log_probs, entropies = [], []
        action_length, action_size = actions_batch.shape
        for i in range(action_length):
            _, entropy, log_prob = self.evaluate(actions_batch[i])
            log_probs.append(log_prob.view(1))
            entropies.append(entropy.view(1))
        return torch.cat(log_probs), torch.cat(entropies)

