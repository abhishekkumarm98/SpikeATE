import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, random_split

class AspectDataset():
    def __init__(self, path, max_seq_length):

        path1 = "/".join(path.split("/")[:-1])
        name = path.split("/")[-1].split(".")[0]

        loaded_dict = np.load(path1+'/'+name+'_augment.npz', allow_pickle=True)['arr_0'].tolist()
        tokenized_reviews = loaded_dict['tokenized_reviews']
        aspect_bio_labels = loaded_dict["bio_format"]

        # Load the saved npz file
        name1 = "Laptops" if "Laptops" in path else "Restaurants"
        word2idx = np.load(path1+'/'+name1+'_word2idx_idx2word'+'_augment.npz', allow_pickle=True)['arr_0'].tolist()['word2idx']

        data = []

        for index, tk_review in enumerate(tokenized_reviews):
            idx = []
            mask = []
            len_cnt = 0
            for tk in tk_review:
                if len_cnt < max_seq_length:
                    idx.append(word2idx[tk])
                    mask.append(1.)
                    len_cnt += 1
                else:
                    break

            source_data_per = (idx + [0] * (max_seq_length - len(idx)))
            source_mask_per = (mask + [0.] * (max_seq_length - len(idx)))

            aspect_label = []
            for l in aspect_bio_labels[index]:
                if l == 'O':
                    aspect_label.append([1, 0, 0])
                elif l == 'B':
                    aspect_label.append([0, 1, 0])
                elif l == 'I':
                    aspect_label.append([0, 0, 1])
                else:
                    raise ValueError

            aspect_y_per = (aspect_label + [[0, 0, 0]] * (max_seq_length - len(idx)))[:max_seq_length]

            data_per = {'x': np.array(source_data_per, dtype='int64'),
                            'mask': np.array(source_mask_per, dtype='float32'),
                            'aspect_y': np.array(aspect_y_per, dtype='int64')}

            data.append(data_per)

        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def convert_to_list(y_aspect, mask):
    y_aspect_list = []
    for seq_aspect, seq_mask in zip(y_aspect, mask):
        l_a = []
        for label_dist_a, m in zip(seq_aspect, seq_mask):
            if m == 0:
                break
            else:
                l_a.append(np.argmax(label_dist_a))

        y_aspect_list.append(l_a)

    return y_aspect_list


def getF1score(true_aspect, predict_aspect):

    begin = 1
    inside = 2
    correct, predicted, relevant = 0, 0, 0

    for i in range(len(true_aspect)):
        true_seq = true_aspect[i]
        predict = predict_aspect[i]

        for num in range(len(true_seq)):
            if true_seq[num] == begin:
                relevant += 1

                if predict[num] == begin:
                    match = True
                    for j in range(num + 1, len(true_seq)):
                        if true_seq[j] == inside and predict[j] == inside:
                            continue
                        elif true_seq[j] != inside and predict[j] != inside:
                            break
                        else:
                            match = False
                            break

                    if match:
                        correct += 1

        for pred in predict:
            if pred == begin:
                predicted += 1

    p_aspect = correct / (predicted + 1e-6)
    r_aspect = correct / (relevant + 1e-6)
    f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

    return f_aspect


def get_metric(y_true_aspect, y_predict_aspect, mask, train_op):

    true_aspect = convert_to_list(y_true_aspect, mask)
    predict_aspect = convert_to_list(y_predict_aspect, mask)

    f_aspect = getF1score(true_aspect, predict_aspect)
    return f_aspect

def evaluation_metrics(data_loader, epoch, model, device, criterion, type_set):
    aspect_loss_per_epoch = 0.
    t_aspect_y_all, t_outputs_all, t_mask_all = list(), list(), list()
    # Evaluation mode
    model.eval()
    with torch.no_grad():
        for t_batch, t_sample_batched in enumerate(data_loader):
            inputs = [t_sample_batched[col].to(device) for col in ['x', 'mask']]
            t_outputs_spk = model(inputs)
            t_aspect_y = t_sample_batched['aspect_y'].float().to(device)
            t_aspect_y_all.extend(t_aspect_y.cpu().tolist())
            t_outputs_all.extend(t_outputs_spk.cpu().tolist())
            t_mask_all.extend(inputs[1].cpu().tolist())
            # Compute loss
            loss = criterion(t_outputs_spk, t_aspect_y)
            aspect_loss_per_epoch += loss.to('cpu').item()

    aspect_loss_per_epoch = aspect_loss_per_epoch / len(data_loader.dataset)
    t_aspect_f1 = get_metric(t_aspect_y_all, t_outputs_all, t_mask_all, 1)
    if (type_set == "Training") or (type_set == "Test"):
        print(f"Epoch: {e+1} | {type_set} Loss: {aspect_loss_per_epoch}")
        print(f'{type_set}: aspect f1={np.round(t_aspect_f1, 4)}')
    else:
        print(f"Epoch: {e+1} | {type_set} Loss: {aspect_loss_per_epoch}")
    return aspect_loss_per_epoch

def reset_mechanism(reset_mechanism_type, spk_nature, w_vdecay, pre_volt, pre_spike, current, vth):
  # reset by subtraction
  if reset_mechanism_type == "subtraction":
    if spk_nature == "ternary":
      volt = w_vdecay * pre_volt - (torch.abs(pre_spike) * vth) + current
      return volt
    else:
      volt = w_vdecay * pre_volt - (pre_spike * vth) + current
      return volt

  # reset to zero
  elif reset_mechanism_type == "zero":
    if spk_nature == "ternary":
      volt = w_vdecay * pre_volt * (1. - torch.abs(pre_spike)) + current
      return volt
    else:
      volt = w_vdecay * pre_volt * (1. - pre_spike) + current
      return volt

  # no reset, pure integration
  elif reset_mechanism_type == "no":
    volt = w_vdecay * pre_volt + current
    return volt

class CurrentBasedLIF(nn.Module):

    def __init__(self, func_v, pseudo_grad_ops, param, data_nature, spk_nature, reset_mechanism_type, is_adaptive_vth):

        """
        args:
        func_v: potential function to produce postsynaptic potential values
        pseudo_grad_ops: pseudo gradient operation
        param: (synaptic current decay, voltage decay, a single scalar voltage threshold or a tensor voltage threshold depending on whether is_adaptive_vth is enabled or not)
        data_nature: data nature (continuous or discrete (spikes))
        spk_nature: spike nature (ternary or binary)
        reset_mechanism_type: reset mechanism (subtraction, zero, no reset)
        is_adaptive_vth: whether to use adaptive thresholding or not for each neuron in the layer
        """

        super(CurrentBasedLIF, self).__init__()
        self.func_v, self.func_v1 = func_v
        self.pseudo_grad_ops = pseudo_grad_ops
        self.w_scdecay, self.w_vdecay, self.vth = param
        self.data_nature = data_nature
        self.spk_nature = spk_nature
        self.reset_mechanism_type = reset_mechanism_type
        self.is_adaptive_vth = is_adaptive_vth


    def forward(self, input_data, state):

        """
        args:
        input_data: input spike event from presynaptic neurons
        state: (output spike of last timestep, current of last timestep, voltage of last timestep)

        return: output spike, (output spike, current, voltage)
        """

        pre_spike, pre_current, pre_volt = state
        if self.data_nature == "cont":
          current = self.w_scdecay * pre_current +  torch.cat((self.func_v(input_data), self.func_v1(input_data)), dim=1)
          volt = reset_mechanism(self.reset_mechanism_type, self.spk_nature, self.w_vdecay, pre_volt, pre_spike, current, self.vth)
        else:
          if self.spk_nature == "ternary":
            ps_input_spike = ((input_data>0).float() - input_data).detach() + input_data
            ns_input_spike = (((input_data<0).float())*(-1.) - input_data).detach() + input_data
            current = self.w_scdecay * pre_current +  self.func_v(ps_input_spike)  + self.func_v1(ns_input_spike)
            volt = reset_mechanism(self.reset_mechanism_type, self.spk_nature, self.w_vdecay, pre_volt, pre_spike, current, self.vth)
          else:
            current = self.w_scdecay * pre_current + self.func_v(input_data)
            volt = reset_mechanism(self.reset_mechanism_type, self.spk_nature, self.w_vdecay, pre_volt, pre_spike, current, self.vth)

        output = self.pseudo_grad_ops(volt, self.vth, self.spk_nature, self.is_adaptive_vth)
        return output, (output, current, volt)



"""
Defining a custom autograd function for computing the gradient
of Heaviside step function using arctan function.

"""
class PseudoGradSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, vth, spk_nature, is_adaptive_vth):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.is_adaptive_vth = is_adaptive_vth
        if spk_nature == "ternary":
          return input.ge(vth).float() + input.le(-vth).float() * (-1.)
        else:
          return input.ge(vth).float()

    # Arctan
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        vth = ctx.vth
        is_adaptive_vth = ctx.is_adaptive_vth
        alpha = 2
        grad = (alpha/ 2 / (1 + (torch.pi / 2 * alpha * input_).pow_(2)) * grad_input)
        if is_adaptive_vth:
          grad_vth = (alpha/ 2 / (1 + (torch.pi / 2 * alpha * vth).pow_(2)) * grad_input)
          return grad, grad_vth, None, None
        else:
          return grad, None, None, None


    # # piecewise_quadratic
    ## @staticmethod
    # def backward(ctx, grad_output):
    #   x_abs = ctx.saved_tensors[0].abs()
    #   alpha=2
    #   mask = (x_abs > (1 / alpha))
    #   grad_x = (grad_output * (- (alpha ** 2) * x_abs + alpha)).masked_fill_(mask, 0)
    #   return grad_x, None, None

    # piecewise_quadratic exp
    ## @staticmethod
    # def backward(ctx, grad_output):
    #   alpha = 2
    #   return  alpha / 2 * (- alpha * ctx.saved_tensors[0].abs()).exp_() * grad_output, None, None

    # # sigmoid
    ## @staticmethod
    # def backward(ctx, grad_output):
    #   alpha = 4
    #   sgax = (ctx.saved_tensors[0] * alpha).sigmoid_()
    #   return grad_output * (1. - sgax) * sgax * alpha, None, None

    # softsign
    ## @staticmethod
    # def backward(ctx, grad_output):
    #   alpha = 2
    #   return grad_output / (2 * alpha * (1 / alpha + ctx.saved_tensors[0].abs()).pow_(2)), None, None

class SpikeATE(nn.Module):

    def __init__(self, numClasses, spike_ts, device, params, path, type_pre_embedding, seq_length, out_features, embed_dim, spike_nature, is_adaptive_vth, reset_mechanism_type):

        """
        args:
        numClasses: Number of Classes
        spike_ts: Number of timesteps
        device: 'cpu' or 'gpu'
        params: (scdecay, vdecay, vth) ; parameters for each LIF neuron
        path: path to load the embeddings
        type_pre_embedding: type of pre-trained embedding (glove, fasttext, word2vec)
        seq_length: sequence length
        out_features: output features (number of channels)
        embed_dim: embedding dimension
        spike_nature: spike nature (ternary or binary)
        is_adaptive_vth: whether to use adaptive thresholding or not for each neuron in the layer
        reset_mechanism_type: reset mechanism (subtraction, zero, no reset)
        """

        super(SpikeATE, self).__init__()
        self.device = device
        self.spike_ts = spike_ts
        self.scdecay, self.vdecay, self.vth = params
        self.path = path
        self.type_pre_embedding = type_pre_embedding
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.out_features = out_features
        self.numClasses = numClasses
        self.spike_nature = spike_nature
        self.is_adaptive_vth = is_adaptive_vth
        self.reset_mechanism_type = reset_mechanism_type

        pseudo_grad_ops = PseudoGradSpike.apply

        # A lookup table to store word embeddings and retrieve them using indices.
        path1 = "/".join(self.path.split("/")[:-1])
        name = "Laptops" if "Laptops" in path else "Restaurants"

        embedding = torch.tensor(np.load(path1+'/'+name+'_'+self.type_pre_embedding+'_embedding_augment'+'.npy'), dtype=torch.float32).to(self.device)

        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True, padding_idx=0)

        # Voltage decay for Convolutional Spike Encoding Layer
        self.conv_spk_enc_w_vdecay = nn.Parameter(torch.ones(1, self.out_features*2, self.seq_length, device=self.device) * self.vdecay)

        # Synaptic current decay for Convolutional Spike Encoding Layer
        self.conv_spk_enc_w_scdecay = nn.Parameter(torch.ones(1, self.out_features*2, self.seq_length, device=self.device) * self.scdecay)

        if self.is_adaptive_vth:
          # Adaptive threshold
          self.conv_spk_enc_vth = nn.Parameter(torch.ones(1, self.out_features*2, self.seq_length,device=self.device) * self.vth)
        else:
          self.conv_spk_enc_vth = self.vth


        # # Voltage decay for Spiking-Based Convolutional Layer1
        self.Spk_conv1_w_vdecay = nn.Parameter(torch.ones(1, self.out_features, self.seq_length, device=self.device) * self.vdecay)

        # Synaptic current decay for Spiking-Based Convolutional Layer1
        self.Spk_conv1_w_scdecay = nn.Parameter(torch.ones(1, self.out_features, self.seq_length, device=self.device) * self.scdecay)

        if self.is_adaptive_vth:
          # Adaptive threshold
          self.Spk_conv1_vth = nn.Parameter(torch.ones(1, self.out_features, self.seq_length,device=self.device) * self.vth)
        else:
          self.Spk_conv1_vth = self.vth


        # Voltage decay for Spiking-Based Convolutional Layer2
        self.Spk_conv2_w_vdecay = nn.Parameter(torch.ones(1, self.out_features, self.seq_length, device=self.device) * self.vdecay)

        # Synaptic current decay for Spiking-Based Convolutional Layer2
        self.Spk_conv2_w_scdecay = nn.Parameter(torch.ones(1, self.out_features, self.seq_length, device=self.device) * self.scdecay)

        if self.is_adaptive_vth:
          # Adaptive threshold
          self.Spk_conv2_vth = nn.Parameter(torch.ones(1, self.out_features, self.seq_length,device=self.device) * self.vth)
        else:
          self.Spk_conv2_vth = self.vth


        # Voltage decay for Spiking-Based Convolutional Layer3
        self.Spk_conv3_w_vdecay = nn.Parameter(torch.ones(1, self.out_features, self.seq_length, device=self.device) * self.vdecay)

        # Synaptic current decay for Spiking-Based Convolutional Layer3
        self.Spk_conv3_w_scdecay = nn.Parameter(torch.ones(1, self.out_features, self.seq_length, device=self.device) * self.scdecay)

        if self.is_adaptive_vth:
          # Adaptive threshold
          self.Spk_conv3_vth = nn.Parameter(torch.ones(1, self.out_features, self.seq_length,device=self.device) * self.vth)
        else:
          self.Spk_conv3_vth = self.vth


        self.conv_spk_enc = CurrentBasedLIF((nn.Conv1d(self.embed_dim, self.out_features, 5, stride=1, padding=2 ,bias=True), nn.Conv1d(self.embed_dim, self.out_features, 5, stride=1, padding=2 ,bias=True)), pseudo_grad_ops,
        [self.conv_spk_enc_w_scdecay, self.conv_spk_enc_w_vdecay, self.conv_spk_enc_vth], "cont", self.spike_nature, self.reset_mechanism_type, self.is_adaptive_vth)

        self.Spk_conv1 = CurrentBasedLIF((nn.Conv1d(self.out_features*2, self.out_features, 5, stride=1, padding=2, bias=True), nn.Conv1d(self.out_features*2, self.out_features, 5, stride=1, padding=2, bias=True)), pseudo_grad_ops,
        [self.Spk_conv1_w_scdecay, self.Spk_conv1_w_vdecay, self.Spk_conv1_vth], "ternary", self.spike_nature, self.reset_mechanism_type, self.is_adaptive_vth)

        self.Spk_conv2 = CurrentBasedLIF((nn.Conv1d(self.out_features, self.out_features, 5, stride=1, padding=2, bias=True), nn.Conv1d(self.out_features, self.out_features, 5, stride=1, padding=2, bias=True)), pseudo_grad_ops,
        [self.Spk_conv2_w_scdecay, self.Spk_conv2_w_vdecay, self.Spk_conv2_vth], "ternary", self.spike_nature, self.reset_mechanism_type, self.is_adaptive_vth)

        self.Spk_conv3 = CurrentBasedLIF((nn.Conv1d(self.out_features, self.out_features, 5, stride=1, padding=2, bias=True), nn.Conv1d(self.out_features, self.out_features, 5, stride=1, padding=2, bias=True)), pseudo_grad_ops,
        [self.Spk_conv3_w_scdecay, self.Spk_conv3_w_vdecay, self.Spk_conv3_vth], "ternary", self.spike_nature, self.reset_mechanism_type, self.is_adaptive_vth)

        self.nonSpk_fc = nn.Linear(self.out_features, self.numClasses)



    def forward(self, input_data, states):

        """
        args:
        input_data: input wafer maps
        states: list of initialized spikes, initialized voltages, initialized currents for each layer

        return: Summation of all output spikes stacked together (sum of probabilities) over all time steps

        """

        output_spikes = []
        conv_spk_enc_state, Spk_conv1_state, Spk_conv2_state, Spk_conv3_state = states[0], states[1], states[2], states[3]

        x_emb = self.embedding(input_data)
        x_emb = x_emb.transpose(1, 2)
        # torch.manual_seed(42)

        for step in range(self.spike_ts):
            conv_spk_enc_spike, conv_spk_enc_state = self.conv_spk_enc(x_emb, conv_spk_enc_state)
            Spk_conv1_spike, Spk_conv1_state = self.Spk_conv1(conv_spk_enc_spike, Spk_conv1_state)
            Spk_conv2_spike, Spk_conv2_state = self.Spk_conv2(Spk_conv1_spike, Spk_conv2_state)
            Spk_conv3_spike, Spk_conv3_state = self.Spk_conv3(Spk_conv2_spike, Spk_conv3_state)
            nonSpk_fc_output = self.nonSpk_fc(Spk_conv3_spike.transpose(1, 2))
            output_spikes += [ F.softmax(nonSpk_fc_output, dim=2) ]

        return torch.stack(output_spikes).sum(dim=0)

class CurrentBasedSNN(nn.Module):

    def __init__(self, numClasses, spike_ts, device, params, path, type_pre_embedding, seq_length, out_features, embed_dim, spike_nature, is_adaptive_vth, reset_mechanism_type):

        super(CurrentBasedSNN, self).__init__()
        self.device = device
        self.seq_length = seq_length
        self.out_features = out_features
        self.spikeate = SpikeATE(numClasses, spike_ts, device, params, path, type_pre_embedding, seq_length, out_features, embed_dim, spike_nature, is_adaptive_vth, reset_mechanism_type)


    def forward(self, input_data):

        """
        args:
        input_data: input word embeddings

        return: Summation of all output spikes stacked together (sum of probabilities) over all time steps
        """

        x, mask = input_data
        batch_size = x.shape[0]

        # Definiing initial States for each spiking layer initialized with a matrix containing zeros.

        # For Convolutional Spike Encoding Layer
        conv_spk_enc_current = torch.zeros(batch_size, self.out_features*2, self.seq_length, device=self.device)
        conv_spk_enc_volt = torch.zeros(batch_size, self.out_features*2, self.seq_length, device=self.device)
        conv_spk_enc_spike = torch.zeros(batch_size, self.out_features*2, self.seq_length, device=self.device)
        conv_spk_enc_state = (conv_spk_enc_spike, conv_spk_enc_current, conv_spk_enc_volt)

        # For Spiking-Based Convolutional Layer1
        Spk_conv1_current = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv1_volt = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv1_spike = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv1_state = (Spk_conv1_spike, Spk_conv1_current, Spk_conv1_volt)

        # For Spiking-Based Convolutional Layer2
        Spk_conv2_current = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv2_volt = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv2_spike = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv2_state = (Spk_conv2_spike, Spk_conv2_current, Spk_conv2_volt)

        # For Spiking-Based Convolutional Layer3
        Spk_conv3_current = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv3_volt = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv3_spike = torch.zeros(batch_size, self.out_features, self.seq_length, device=self.device)
        Spk_conv3_state = (Spk_conv3_spike, Spk_conv3_current, Spk_conv3_volt)

        states = (conv_spk_enc_state, Spk_conv1_state, Spk_conv2_state, Spk_conv3_state)

        output = self.spikeate(x, states)

        return output


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Token Level classification for aspect term extraction task.')
    parser.add_argument('--epoch', type = int, help = "Number of epochs.", default = 18)
    parser.add_argument('--batchSize', type = int, help = "Batch Size.", default = 8)
    parser.add_argument('--learningRate', type = float, help = "Learning Rate.", default = 0.0001) 
    parser.add_argument('--numCls', type = int, help = "Number of classes.", default = 3)
    parser.add_argument('--numTimeSteps', type = int, help = "Number of time steps", default = 5)
    parser.add_argument('--vth', type = float, help = "Threshold voltage", default = 0.1)
    parser.add_argument('--scdecay', type = float, help = "Synaptic current decay", default = 0.1)
    parser.add_argument('--vdecay', type = float, help = "Voltage decay", default = 0.1)
    parser.add_argument('--isadapVth', type = int, help = "Adaptive threshold volatge", default = 0)
    parser.add_argument('--seqLen', type = int, help = "Sequence length", default = 82)
    parser.add_argument('--embedDim', type = int, help = "Embedding dimension of a word", default = 300)
    parser.add_argument('--numChn', type = int, help = "Number of output channels", default = 256)
    parser.add_argument('--numValSamples', type = int, help = "Number of validation samples", default = 150)
    parser.add_argument('--spkNature', type = str, help = "Spike nature: binary or ternary", default = 'ternary')
    parser.add_argument('--resetMechanism', type = str, help = "Reset mechanism type: no, zero, subtraction", default = 'zero')
    parser.add_argument('--preembeddingType', type = str, help = "Pre-embedding type: glove, fasttext, word2vec", default = 'glove')
    parser.add_argument('--trainDataPath', type = str, help = "Training data path", default = '')
    parser.add_argument('--testDataPath', type = str, help = "Test data path", default = '')
    parser.add_argument('--seedNum', type = int, help = "Seed number", default = 42)
    parser.add_argument('--numRuns', type = int, help = "Number of runs", default = 3)
    
    args = parser.parse_args()

    # Training hyperparameters
    num_epochs = args.epoch
    batch_size = args.batchSize
    learning_rate = args.learningRate
    num_outputs = args.numCls
    num_steps = args.numTimeSteps
    VTH = args.vth
    SCDECAY = args.scdecay
    VDECAY = args.vdecay
    spike_nature = args.spkNature
    is_adaptive_vth = args.isadapVth
    reset_mechanism_type = args.resetMechanism
    params = [SCDECAY, VDECAY, VTH]
    seq_length = args.seqLen
    embed_dim = args.embedDim
    out_features = args.numChn
    valset_num = args.numValSamples
    type_pre_embedding = args.preembeddingType
    train_data_path = args.trainDataPath
    test_data_path = args.testDataPath
    runs = args.numRuns
    seed_num = args.seedNum
    
    for run in range(runs):
        print("Run number:", run)
        print("Number of pochs:", num_epochs)
        print("Batch size:", batch_size)
        print("Learning rate:", learning_rate)
        print("Number of classes:", num_outputs)
        print("Time steps:", num_steps)
        print("Scdecay, Vdecay, Vth:", params)
        print("Spike nature (binary or ternary):", spike_nature)
        print("Vth is adaptive or not:", is_adaptive_vth)
        print("Reset mechanism type:", reset_mechanism_type)
        print("Sequence length:", seq_length)
        print("Embedding dimension:", embed_dim)
        print("Number of output channels:",out_features)
        print("Number of validation samples:", valset_num)
        print("Type of pre_embedding:", type_pre_embedding)
        print("Seed num", seed_num+run)
        print("Number of runs:", runs)
        print("Training path:", train_data_path)
        print("Test path:", test_data_path)
        print()
        
        seed_num = seed_num + run
        # For reproducibility
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        np.random.seed(seed_num)
        random.seed(seed_num)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
        # Detecting a device (CPU or GPU)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Device: {device} \n")
    
        trainset = AspectDataset(train_data_path, max_seq_length = seq_length)
        testset = AspectDataset(test_data_path, max_seq_length = seq_length)
    
        if valset_num > 0:
            trainset, valset = random_split(trainset, (len(trainset)- valset_num, valset_num))
        else:
            valset = testset
    
        # Instantiate a DataLoader object
        train_data_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=valset, batch_size=len(valset), shuffle=False)
        test_data_loader = DataLoader(dataset=testset, batch_size=len(testset), shuffle=False)
    
        print(f"Total number of training samples: {len(train_data_loader.dataset)}")
        print(f"Total number of validation samples: {len(val_data_loader.dataset)}")
        print(f"Total number of test samples: {len(test_data_loader.dataset)} \n")
    
        # Instantiate and load the SNN model onto GPU (CUDA), if available else CPU
        model = CurrentBasedSNN(num_outputs, num_steps, device, params, train_data_path, type_pre_embedding, seq_length, out_features, embed_dim, spike_nature, is_adaptive_vth, reset_mechanism_type).to(device)
        print("\nSNN Model Summary:\n")
        print(model)
        print()
    
        # Cross entropy loss
        criterion = nn.CrossEntropyLoss()
    
        # Scaler
        scaler = torch.amp.GradScaler(device.type)
    
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay = 1e-5)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay = 1e-5)
        # optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9, weight_decay = 1e-5)
    
        from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, threshold=0.01, threshold_mode='abs')
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    
        # Training
        for e in range(num_epochs):
            model.train() # Training mode
            for i_batch, sample_batched in enumerate(train_data_loader):
                inputs = [sample_batched[col].to(device) for col in ['x', 'mask']]
                t_aspect_y = sample_batched['aspect_y'].float().to(device)
                # Flush the initial gradients
                optimizer.zero_grad()
                # Forward passs
                with torch.amp.autocast(device.type):
                  output_spk = model(inputs)
                  loss = criterion(output_spk, t_aspect_y)
    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    
            train_loss = evaluation_metrics(train_data_loader, e, model, device, criterion, type_set="Training")
            val_loss = evaluation_metrics(val_data_loader, e, model, device, criterion, type_set="Validation")
            # Step the scheduler
            scheduler.step()
            print()
    
        # Evaluation
        evaluation_metrics(test_data_loader, e, model, device, criterion, type_set="Test")
    
    
        name = model.__class__.__name__
        path = os.getcwd()+"/"+"Models/"
    
        if not os.path.exists(path):
            os.makedirs(path)
    
        # Save the SNN model
        torch.save(model.state_dict(), path + name + "_SemEval_run_" + str(run) + ".pth")
        print("\n\n")





