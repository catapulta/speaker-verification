"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import loader


def training_routine(net, n_iters, lr, gpu, train_loader, val_loader, layer_name, embedding_size):
    gpu = gpu and torch.cuda.is_available()
    if not gpu:
        print('Not using GPU.')
    import logging
    logging.basicConfig(filename='train.log', level=logging.DEBUG)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)

    # switch to train mode
    net.train()

    for i in range(n_iters):
        tic = time.time()
        train_prediction = []
        train_observed = []
        for j, (train_labels, train_data) in enumerate(train_loader):
            if gpu:
                train_labels, train_data, net = train_labels.cuda(), train_data.cuda(), net.cuda()
            # forward pass
            train_output = net(train_data)
            train_loss = criterion(train_output, train_labels)
            # backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            scheduler.step()
            optimizer.step()
            train_output = train_output.cpu().argmax(dim=1).detach().numpy()
            train_prediction.append(train_output)
            train_labels = np.array(train_labels.cpu().numpy())
            train_observed.append(train_labels)
            torch.cuda.empty_cache()

            # Training print
            if j % 2 == 0:
                t = 'At {:.0f}% of epoch {}'.format(
                    j * train_loader.batch_size / train_loader.dataset.num_entries * 100, i)
                print(t)
                logging.info(t)
                train_accuracy = np.array(train_output == train_labels).mean()
                t = "Training loss : {}".format(train_loss.cpu().detach().numpy())
                print(t)
                logging.info(t)
                t = "Training accuracy {}:".format(train_accuracy)
                print(t)
                logging.info(t)
                t = '--------------------------------------------'
                print(t)
                logging.info(t)

        # Once every 1 epochs, print validation statistics
        torch.save(net, 'model.torch')
        epochs_print = 1
        if i % epochs_print == 0:
            with torch.no_grad():
                t = "#########  Epoch {} #########".format(i)
                print(t)
                logging.info(t)
                # compute the accuracy of the prediction
                train_prediction = np.concatenate(train_prediction)
                train_observed = np.concatenate(train_observed)
                train_accuracy = (train_prediction == train_observed).mean()
                # Now for the validation set
                val_prediction = []
                val_observed = []
                for j, (val_labels, val_enrol, val_test) in (enumerate(val_loader)):
                    if gpu:
                        val_labels, val_data, val_test = val_labels.cuda(), val_enrol.cuda(), val_test.cuda()
                    embedding1 = extract_embedding(val_enrol, net, layer_name, (len(val_enrol), embedding_size))
                    embedding2 = extract_embedding(val_test, net, layer_name, (len(val_enrol), embedding_size))
                    cos = torch.nn.CosineSimilarity()
                    val_output = cos(embedding1, embedding2)
                    val_prediction.append(val_output)
                    val_labels = val_labels.cpu().numpy()
                    val_observed.append(val_labels)
                    if j>40:
                        break
                val_prediction = np.concatenate(val_prediction)
                val_observed = np.concatenate(val_observed)
                # compute the accuracy of the prediction
                val_eed = utils.EER(val_observed, val_prediction)
                t = "Training accuracy : {}".format(train_accuracy)
                print(t)
                logging.info(t)
                t = "Validation EER {}:".format(val_eed)
                print(t)
                logging.info(t)
                toc = time.time()
                t = "Took: {}".format((toc - tic) / epochs_print)
                print(t)
                logging.info(t)
                t = '--------------------------------------------'
                print(t)
                logging.info(t)

    net = net.cpu()
    return net


def train_net(net, layer_name, embedding_size, lr=0.05, n_iters=350, batch_size=100, num_workers=4):
    train_dataset = loader.UtteranceTrainDataset()

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    val_dataset = loader.UtteranceValidationDataset()
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    net = net(train_loader.dataset.n_labels)
    net = training_routine(net, n_iters, lr, True, train_loader, val_loader, layer_name, embedding_size)
    return net


def extract_embedding(x, net, layer_name, embedding_size):
    # function to copy the tensor inside the layer
    def get_embedding(self, input, output):
        embedding.copy_(output.data)

    # get the layer
    layer = net._modules.get(layer_name)
    # instantiate embedding container
    embedding = torch.zeros(embedding_size)
    # get the layer
    layer = layer.register_forward_hook(get_embedding)
    # forward pass
    net(x)
    # remove the hook
    layer.remove()
    return embedding


def infer_embeddings(net, layer_name, embedding_size, transform=False, gpu=True):
    test_dataset = loader.UtteranceDataset(data_type='test', transform=transform)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=100,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    t = 'Performing inference...'
    print(t)

    gpu = gpu and torch.cuda.is_available()
    if not gpu:
        print('Not using GPU for testing.')
    with torch.no_grad():
        test_prediction = []
        test_observed = []
        for j, (test_labels, test_enrol, test_test) in tqdm(enumerate(test_loader)):
            if gpu:
                test_labels, test_data, test_test = test_labels.cuda(), test_enrol.cuda(), test_test.cuda()
            embedding1 = extract_embedding(test_enrol, net, layer_name, (len(test_enrol), embedding_size))
            embedding2 = extract_embedding(test_test, net, layer_name, (len(test_enrol), embedding_size))
            cos = torch.nn.CosineSimilarity()
            test_output = cos(embedding1, embedding2)
            test_prediction.append(test_output)
            test_labels = test_labels.cpu().numpy()
            test_observed.append(test_labels)
        test_prediction = np.concatenate(test_prediction)
        test_observed = np.concatenate(test_observed)
        # compute the accuracy of the prediction
        test_eed = utils.EER(test_observed, test_prediction)

    return test_eed


def infer_net(net, test_loader, gpu):
    gpu = gpu and torch.cuda.is_available()
    if not gpu:
        print('Not using GPU for testing.')
    with torch.no_grad():
        # compute the accuracy of the prediction
        # Now for the validation set
        test_prediction = []
        for j, test_data in enumerate(test_loader):
            if gpu:
                test_data = test_data.cuda()
            test_output = net(test_data).cpu().argmax(dim=1).detach().numpy()
            test_prediction.append(test_output)
        test_prediction = np.concatenate(test_prediction)
        return test_prediction


def get_predictions(net, transform=False):
    test_dataset = loader.UtteranceDataset(data_type='test', transform=transform)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=100,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    t = 'Performing inference...'
    print(t)

    test_prediction = infer_net(net, test_loader, gpu=True)
    return test_prediction


def write_results(predictions, output_file='predictions.txt'):
    """
    Write predictions to file for submission.
    File should be:
        named 'predictions.txt'
        in the root of your tar file
    :param predictions: iterable of integers
    :param output_file:  path to output file.
    :return: None
    """
    print('Printing results to file...')
    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y))


def xavier_init(model):
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
        if hasattr(module, 'bias'):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
    return model


if __name__ == '__main__':
    import model
    import utils

    # all_cnn = train_net(net=model.all_cnn_module, lr=0.1, n_iters=350, batch_size=150, num_workers=4)
    all_cnn = train_net(layer_name='24', embedding_size=100, net=model.all_cnn_module, lr=0.005, n_iters=1,
                        batch_size=20, num_workers=1)
    # write_results(test_prediction)
