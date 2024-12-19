import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer.Models import get_non_pad_mask, get_non_ex_mask
from transformer import Constants


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask, non_ex_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)
    event.masked_fill_(~non_ex_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask, non_ex_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = ((time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]) * non_ex_mask[:, 1:]
    diff_lambda = ((all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]) * non_ex_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, non_ex_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = ((time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]) * non_ex_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device) 
    # dim : (sz_b) * (seq_len - 1) * (num_samples)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    
    # dim : (sz_b) * (seq_len - 1) * (num_samples) * (k)
    temp_time = einops.repeat(temp_time, 'sz_b seq_len num_samples -> sz_b seq_len num_samples k', k=model.num_types)
    
    # dim : (sz_b) * (seq_len - 1) * (K)
    temp_hid = model.linear(data)[:, 1:, :]
    
    temp_hid = einops.repeat(temp_hid, 'sz_b seq_len k -> sz_b seq_len num_samples k', num_samples=num_samples)
    
    # we now have lambdas for (num_samples * k) combinations
    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    
    ex_type_mask = torch.ones([*all_lambda.size()], device=data.device)
    for ex_type in Constants.event_types_moodle_ex:
        ex_type_mask[:, :, :, ex_type - 1] = torch.zeros([*all_lambda.size()[:-1]]).to(data.device)
        

    # lambda = Sigma lambda_k for all event types
    all_lambda = torch.sum(all_lambda*ex_type_mask, dim=3)
    
    # Sigma lambda over all sampled points
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence. """
    
    non_pad_mask = get_non_pad_mask(types).squeeze(2)
    non_ex_mask = get_non_ex_mask(types).squeeze(2)
    
    ex_type_mask = torch.ones([*types.size(), model.num_types], device=data.device)
    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)
    for ex_type in Constants.event_types_moodle_ex:
        ex_type_mask[:, :, ex_type - 1] = torch.zeros([*types.size()]).to(data.device)
        
    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    # Removed the type mask, add lambda_k for all k, not just the occurence type
    type_lambda = torch.sum(all_lambda * ex_type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask, non_ex_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, non_ex_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func, num_types):
    """ Event prediction loss, cross entropy or label smoothing. """

    non_ex_mask = get_non_ex_mask(types).squeeze(2)[:,1:]
    
    type_mask = torch.ones([*types[:, 1:].size(), num_types], device=types.device)
    for ex_type in Constants.event_types_moodle_ex:
        type_mask[:, :, ex_type - 1] = torch.zeros([*types[:, 1:].size()], device=types.device)
    
    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = (types[:, 1:] * non_ex_mask) - 1
    prediction = prediction[:, :-1, :] * type_mask
    
    pred_type = torch.max(prediction, dim=-1)[1]
    
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time, types):
    """ Time prediction loss. """

    non_ex_mask = get_non_ex_mask(types).squeeze(2)[:,1:]
    
    prediction.squeeze_(-1)
    
    true = event_time[:, 1:] - event_time[:, :-1]

    prediction = prediction[:, :-1]
    prediction = torch.where(true >= 0., prediction, torch.zeros(true.shape).to(torch.device("cuda")))
    prediction = prediction * non_ex_mask
    
    true = torch.where(true >= 0., true, torch.zeros(true.shape).to(torch.device("cuda")))
    true = true * non_ex_mask

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


def splitted_type_loss(prediction, types, loss_func, num_types):
    max_len = len(types[0])
    all_test_sequence_event_type = []
    all_predicted_prob_event_type = []

    batch, seq_length = types.shape

    for event_type_index in range(len(types)):

        actual_event_time = types[event_type_index].cpu().numpy()

        try:
            original_zero_index = list(actual_event_time).index(0)

        except:
            original_zero_index = len(actual_event_time)

        non_zero_test_event_length = int(len(actual_event_time[:original_zero_index]) * Constants.split_ratio)
        test_event_type = types[event_type_index][non_zero_test_event_length:original_zero_index]
        test_event_type = list(test_event_type) + [Constants.PAD] * (max_len - len(test_event_type))

        all_test_sequence_event_type.append(test_event_type)

        predicted_test = prediction[event_type_index, non_zero_test_event_length:original_zero_index, :]
        seq_length, total_type = predicted_test.shape
        padding_zeros = torch.zeros(max_len - seq_length, total_type).to(torch.device('cuda'))
        padded_predicted = torch.cat([predicted_test, padding_zeros], dim=0)

        all_predicted_prob_event_type.append(padded_predicted)

    types = torch.tensor(all_test_sequence_event_type).to(torch.device('cuda'))
       
    result = torch.cat(all_predicted_prob_event_type, dim=0)
    
    _, total_num_type = result.shape
    prediction = result.reshape(batch, -1, total_num_type)
    
    """ Event prediction loss, cross entropy or label smoothing. """

    non_ex_mask = get_non_ex_mask(types).squeeze(2)[:,1:]
    
    type_mask = torch.ones([*types[:, 1:].size(), num_types], device=types.device)
    for ex_type in Constants.event_types_moodle_ex:
        type_mask[:, :, ex_type - 1] = torch.zeros([*types[:, 1:].size()], device=types.device)
    
    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = (types[:, 1:] * non_ex_mask) - 1
    prediction = prediction[:, :-1, :] * type_mask

    pred_type = torch.max(prediction, dim=-1)[1]

    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)

    return types, loss, correct_num


def split_tensor(prediction, event_time):
    max_len = len(event_time[0])
    all_test_sequence = []
    all_predicted_sequence = []
    for event_time_index in range(len(event_time)):

        actual_event_time = event_time[event_time_index].cpu().numpy()
        try:
            #original_zero_index = list(actual_event_time).index(0)
            original_zero_index = list(actual_event_time[1:]).index(0) + 1
        except:
            original_zero_index = len(actual_event_time)

        non_zero_test_event_length = int(len(actual_event_time[:original_zero_index]) * Constants.split_ratio)
        test_event_time = event_time[event_time_index][non_zero_test_event_length:original_zero_index]

        test_event_time = list(test_event_time) + [Constants.PAD] * (max_len - len(test_event_time))

        all_test_sequence.append(test_event_time)
        predicted = prediction[event_time_index].view(1, -1)[0].cpu().numpy()
        predicted_time = predicted[non_zero_test_event_length:original_zero_index]
        predicted_event_time = list(predicted_time) + [Constants.PAD] * (max_len - len(predicted_time))
        all_predicted_sequence.append(predicted_event_time)

    converted_tensor = torch.tensor(all_test_sequence).to(torch.device('cuda'))
    predicted_event_tensor = torch.tensor(all_predicted_sequence).to(torch.device('cuda'))

    return converted_tensor, predicted_event_tensor


def calculate_base_time_loss(event_time):
    batch_se = 0
    pred_length = 0
    for event_time_index in range(len(event_time)):

        actual_event_time = event_time[event_time_index].numpy()
        try:
            original_zero_index = list(actual_event_time).index(0)
        except:
            original_zero_index = len(actual_event_time)

        non_zero_test_event_length = int(len(actual_event_time[:original_zero_index]) * Constants.split_ratio)
        original_event_time = actual_event_time[:original_zero_index]
        original_event_time_gap = original_event_time[1:] - original_event_time[:-1]

        predicted_time_gap = []
        for event_time_index in range(len(original_event_time_gap)):

            if event_time_index > 0:
                mean_time_gap = np.mean(original_event_time_gap[:event_time_index])
            else:
                mean_time_gap = original_event_time_gap[event_time_index]

            predicted_time_gap.append(mean_time_gap)

        actual_test_time_gap = original_event_time_gap[non_zero_test_event_length:]
        predicted_test_time_gap = predicted_time_gap[non_zero_test_event_length:]

        pred_length = pred_length + len(predicted_test_time_gap)
        # calculate squared sum
        diff = predicted_test_time_gap - actual_test_time_gap
        se = np.sum(diff * diff)
        batch_se = batch_se + se

    return batch_se, pred_length


def calculate_base_type_loss_personalized(event_type):
    batch_se = 0
    pred_length = 0
    total_correct = 0

    for event_time_index in range(len(event_type)):

        actual_event_time = event_type[event_time_index].numpy()
        try:
            original_zero_index = list(actual_event_time).index(0)
        except:
            original_zero_index = len(actual_event_time)

        non_zero_test_event_length = int(len(actual_event_time[:original_zero_index]) * 0.70)
        original_event_time = actual_event_time[:original_zero_index]

        original_types = list(original_event_time[non_zero_test_event_length:])
        frequent_type = mode(original_types)
        predicted_types = [frequent_type] * len(original_types)
        pred_length = pred_length + len(predicted_types)

        events_correct = len(
            [predicted_types[i] for i in range(0, len(predicted_types)) if predicted_types[i] == original_types[i]])

        total_correct = total_correct + events_correct
    return total_correct, pred_length


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
