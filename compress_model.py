#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import BayesianLayers
from compression import compute_compression_rate, compute_reduced_weights
from utils import visualize_pixel_importance, generate_gif, visualise_weights

"""
This portion of the code is based in the 'example.py' file of: 
Karen Ullrich, Christos Louizos, Oct 2017
"""

def compress_model(injection_values, measurement_vector, train_frac, network_state_samples, layer_proportions, N, FLAGS):
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}
    
    train_tensor_x = torch.Tensor(measurement_vector[:int(measurement_vector.shape[0]*train_frac)]).unsqueeze(1) # transform to torch tensor
    train_tensor_y = torch.Tensor(network_state_samples[:int(measurement_vector.shape[0]*train_frac)])#.unsqueeze(1)

    train_dataset = torch.utils.data.TensorDataset(train_tensor_x,train_tensor_y) # create your datset
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=FLAGS.batchsize, shuffle=True, **kwargs) # create your dataloader

    print(train_tensor_x.shape,train_tensor_y.shape)
    
    
    test_tensor_x = torch.Tensor(measurement_vector[int(measurement_vector.shape[0]*train_frac):]).unsqueeze(1) # transform to torch tensor
    test_tensor_y = torch.Tensor(network_state_samples[int(measurement_vector.shape[0]*train_frac):])#.unsqueeze(1)

    test_dataset = torch.utils.data.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=FLAGS.batchsize, shuffle=True, **kwargs) # create your dataloader

    # for later analysis we take some sample digits
    mask = 255. * (np.ones((1, 1, injection_values[0].shape[0])))

    # build a simple MLP
    class Net(nn.Module):
        def __init__(self, layer_proportions):
            super(Net, self).__init__()
            # activation
            self.relu = nn.ReLU()
            # layers
            self.fc1 = BayesianLayers.LinearGroupNJ(measurement_vector.shape[1], layer_proportions[0]*measurement_vector.shape[1], clip_var=0.04, cuda=FLAGS.cuda)
            self.fc2 = BayesianLayers.LinearGroupNJ(layer_proportions[0]*measurement_vector.shape[1],layer_proportions[1]*measurement_vector.shape[1], cuda=FLAGS.cuda)
            self.fc3 = BayesianLayers.LinearGroupNJ(layer_proportions[1]*measurement_vector.shape[1],network_state_samples.shape[1], cuda=FLAGS.cuda)
            # layers including kl_divergence
            self.kl_list = [self.fc1, self.fc2, self.fc3]

        def forward(self, x):
            x = x.view(-1, measurement_vector.shape[1])#28 * 28)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

        def get_masks(self,thresholds):
            weight_masks = []
            mask = None
            for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
                # compute dropout mask
                if mask is None:
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                else:
                    mask = np.copy(next_mask)
                try:
                    log_alpha = layers[i + 1].get_log_dropout_rates().cpu().data.numpy()
                    next_mask = log_alpha < thresholds[i + 1]
                except:
                    # must be the last mask
                    next_mask = np.ones(network_state_samples.shape[1])

                weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
                weight_masks.append(weight_mask.astype(np.float))
            return weight_masks

        def kl_divergence(self):
            KLD = 0
            for layer in self.kl_list:
                KLD += layer.kl_divergence()
            return KLD

    # init model
    model = Net(layer_proportions)
    if FLAGS.cuda:
        model.cuda()

    # init optimizer
    optimizer = optim.Adam(model.parameters())

    # we optimize the variational lower bound scaled by the number of data
    # points (so we can keep our intuitions about hyper-params such as the learning rate)
    discrimination_loss = nn.functional.mse_loss #cross_entropy

    def objective(output, target, kl_divergence):
        discrimination_error = discrimination_loss(output, target)
        variational_bound = discrimination_error + kl_divergence / N
        if FLAGS.cuda:
            variational_bound = variational_bound.cuda()
        return variational_bound

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            #print("output: ", output.shape,"target: ", target.squeeze().shape)
            loss = objective(output, target, model.kl_divergence())
            loss.backward()
            optimizer.step()
            # clip the variances after each step
            for layer in model.kl_list:
                layer.clip_variances()
        print('Epoch: {} \tTrain loss: {:.8f} \t'.format(
            epoch, loss.data))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += discrimination_loss(output, target, size_average=False).data
            pred = output.data#.max(1, keepdim=True)[1]
            #print("pred: ", pred.shape)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('Test loss: {:.8f}, Accuracy: {}/{} ({:.8f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # train the model and save some visualisations on the way
    for epoch in range(1, FLAGS.epochs + 1):
        train(epoch)
        test()
        # visualizations
        weight_mus = [model.fc1.weight_mu, model.fc2.weight_mu]
        log_alphas = [model.fc1.get_log_dropout_rates(), model.fc2.get_log_dropout_rates(),
                      model.fc3.get_log_dropout_rates()]
        visualise_weights(weight_mus, log_alphas, epoch=epoch)
        log_alpha = model.fc1.get_log_dropout_rates().cpu().data.numpy()
        #visualize_pixel_importance(images, log_alpha=log_alpha, epoch=str(epoch))

#     generate_gif(save='pixel', epochs=FLAGS.epochs)
#     generate_gif(save='weight0_e', epochs=FLAGS.epochs)
#     generate_gif(save='weight1_e', epochs=FLAGS.epochs)

    # compute compression rate and new model accuracy
    layers = [model.fc1, model.fc2, model.fc3]
    thresholds = FLAGS.thresholds
    #print(model.get_masks(thresholds))
    compute_compression_rate(layers, model.get_masks(thresholds))

    print("Test error after with reduced bit precision:")

    weights = compute_reduced_weights(layers, model.get_masks(thresholds))
    for layer, weight in zip(layers, weights):
        if FLAGS.cuda:
            layer.post_weight_mu.data = torch.Tensor(weight).cuda()
        else:
            layer.post_weight_mu.data = torch.Tensor(weight)
    for layer in layers: layer.deterministic = True
    test()

