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
import net_sphere
import utils


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def training_routine(net, n_epochs, lr, gpu, train_loader, val_loader, layer_name, embedding_size):
    gpu = gpu and torch.cuda.is_available()
    if not gpu:
        print('Not using GPU.')
    import logging
    logging.basicConfig(filename='train.log', level=logging.DEBUG)
    # criterion = nn.CrossEntropyLoss()
    criterion = net_sphere.AngleLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)
    if gpu:
        net.cuda()

    # switch to train mode
    net.train()
    best_rate = 100
    for i in range(n_epochs):
        tic = time.time()
        train_prediction = []
        train_observed = []
        for j, (train_labels, train_data) in enumerate(train_loader):
            if gpu:
                train_labels, train_data = train_labels.cuda(), train_data.cuda()
            # forward pass
            train_output = net(train_data)
            train_loss = criterion(train_output, train_labels)
            # backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # train_output = train_output.cpu().argmax(dim=1).detach().numpy()
            train_prediction.append(train_output)
            train_labels = np.array(train_labels.cpu().numpy())
            train_observed.append(train_labels)
            torch.cuda.empty_cache()

            # training print
            if j % 4 == 0 and j != 0:
                t = 'At {:.0f}% of epoch {}'.format(
                    j * train_loader.batch_size / train_loader.dataset.num_entries * 100, i)
                print(t)
                logging.info(t)
                #    train_accuracy = np.array(train_output == train_labels).mean()
                t = "Training loss : {}".format(train_loss.cpu().detach().numpy())
                print(t)
                logging.info(t)
                #    t = "Training accuracy {}:".format(train_accuracy)
                #    print(t)
                #    logging.info(t)
                t = '--------------------------------------------'
                print(t)
                logging.info(t)

        scheduler.step()
        # every 1 epochs, print validation statistics
        epochs_print = 10
        if i % epochs_print == 0 and not i == 0:
            with torch.no_grad():
                t = "#########  Epoch {} #########".format(i)
                print(t)
                logging.info(t)
                # compute the accuracy of the prediction
                # train_prediction = np.concatenate(train_prediction)
                # train_observed = np.concatenate(train_observed)
                # train_accuracy = (train_prediction == train_observed).mean()
                # Now for the validation set
                val_prediction = []
                val_observed = []
                enrol = {}
                test = {}
                for j, (trial, val_labels, val_enrol, val_test) in (enumerate(val_loader)):
                    if gpu:
                        val_labels, val_enrol, val_test = val_labels.cuda(), val_enrol.cuda(), val_test.cuda()
                    key_enrol_array, key_test_array = trial[:, 0], trial[:, 1]
                    embedding_test = []
                    embedding_enrol = []
                    for t in range(len(key_enrol_array)):
                        key_enrol = key_enrol_array[t]
                        key_test = key_test_array[t]
                        if key_test not in test:
                            test[key_test] = extract_embedding(val_test[t].unsqueeze(0), net, layer_name,
                                                               (len(val_test[t]), embedding_size))
                        embedding_test.append(test[key_test])
                        if key_enrol not in enrol:
                            enrol[key_enrol] = extract_embedding(val_enrol[t].unsqueeze(0), net, layer_name,
                                                                 (len(val_enrol[t]), embedding_size))
                        embedding_enrol.append(enrol[key_enrol])
                    embedding_enrol = torch.cat(embedding_enrol)
                    embedding_test = torch.cat(embedding_test)
                    cos = torch.nn.CosineSimilarity()
                    val_output = cos(embedding_test, embedding_enrol)
                    val_prediction.append(val_output)
                    val_labels = val_labels.cpu().numpy()
                    val_observed.append(val_labels)
                val_prediction = np.concatenate(val_prediction)
                val_observed = np.concatenate(val_observed)
                #  compute the accuracy of the prediction
                val_eed = utils.EER(val_observed, val_prediction)
                # t = "Training accuracy : {}".format(train_accuracy)
                # print(t)
                # logging.info(t)
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
                if best_rate > val_eed[0]:
                    save_model(net, 'model.torch')
                    best_rate = val_eed[0]

    net = net.cpu()
    return net


def train_net(net, layer_name, embedding_size, utterance_size, parts, pretrained_path=None, lr=0.05, n_epochs=350,
              batch_size=100, num_workers=4):
    train_dataset = loader.UtteranceTrainDataset(parts=parts, utterance_size=utterance_size)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    val_dataset = loader.UtteranceValidationDataset(utterance_size=utterance_size)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    net = net(train_loader.dataset.n_labels)
    if pretrained_path is not None:
        pretrained_dict = torch.load(pretrained_path)
        net = load_my_state_dict(net, pretrained_dict)
        print('Loaded pre-trained weights.')
    else:
        net = xavier_init(net)
    net = training_routine(net, n_epochs, lr, True, train_loader, val_loader, layer_name, embedding_size)
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


def infer_embeddings(net, layer_name, embedding_size, utterance_size, gpu=True):
    test_dataset = loader.UtteranceTestDataset(utterance_size=utterance_size)

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
        enrol = {}
        test = {}
        for j, (trial, test_enrol, test_test) in (enumerate(test_loader)):
            if gpu:
                test_enrol, test_test = test_enrol.cuda(), test_test.cuda()
            key_enrol_array, key_test_array = trial[:, 0], trial[:, 1]
            embedding_test = []
            embedding_enrol = []
            for t in range(len(key_enrol_array)):
                key_enrol = key_enrol_array[t]
                key_test = key_test_array[t]
                if key_test not in test:
                    test[key_test] = extract_embedding(test_test[t].unsqueeze(0), net, layer_name,
                                                       (len(test_test[t]), embedding_size))
                embedding_test.append(test[key_test])
                if key_enrol not in enrol:
                    enrol[key_enrol] = extract_embedding(test_enrol[t].unsqueeze(0), net, layer_name,
                                                         (len(test_enrol[t]), embedding_size))
                embedding_enrol.append(enrol[key_enrol])
            embedding_enrol = torch.cat(embedding_enrol)
            embedding_test = torch.cat(embedding_test)
            cos = torch.nn.CosineSimilarity()
            test_output = cos(embedding_test, embedding_enrol)
            test_prediction.append(test_output.cpu().numpy())
        test_prediction = np.concatenate(test_prediction)
        # compute the accuracy of the prediction

    return test_prediction


def infer_net(net, test_loader, gpu):
    gpu = gpu and torch.cuda.is_available()
    if not gpu:
        print('Not using GPU for testing.')
    with torch.no_grad():
        test_prediction = []
        for j, test_data in enumerate(test_loader):
            if gpu:
                net.cuda()
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


def validate(net, layer_name, embedding_size, batch_size, num_workers, utterance_size, gpu):
    gpu = gpu and torch.cuda.is_available()
    if not gpu:
        print('Not using GPU.')
    else:
        net.cuda()
    val_dataset = loader.UtteranceValidationDataset(utterance_size=utterance_size)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    val_prediction = []
    val_observed = []
    enrol = {}
    test = {}
    for j, (trial, val_labels, val_enrol, val_test) in (enumerate(val_loader)):
        if gpu:
            val_enrol, val_test = val_enrol.cuda(), val_test.cuda()
        key_enrol_array, key_test_array = trial[:, 0], trial[:, 1]
        embedding_test = []
        embedding_enrol = []
        for t in range(len(key_enrol_array)):
            key_enrol = key_enrol_array[t]
            key_test = key_test_array[t]
            if key_test not in test:
                test[key_test] = extract_embedding(val_test[t].unsqueeze(0), net, layer_name,
                                                   (len(val_test[t]), embedding_size))
            embedding_test.append(test[key_test])
            if key_enrol not in enrol:
                enrol[key_enrol] = extract_embedding(val_enrol[t].unsqueeze(0), net, layer_name,
                                                     (len(val_enrol[t]), embedding_size))
            embedding_enrol.append(enrol[key_enrol])
        embedding_enrol = torch.cat(embedding_enrol)
        embedding_test = torch.cat(embedding_test)
        cos = torch.nn.CosineSimilarity()
        val_output = cos(embedding_test, embedding_enrol)
        val_prediction.append(val_output.cpu().numpy())
        val_observed.append(val_labels.numpy())
    val_prediction = np.concatenate(val_prediction)
    val_observed = np.concatenate(val_observed)
    #  compute the accuracy of the prediction
    val_eed = utils.EER(val_observed, val_prediction)
    t = "Validation EER {}:".format(val_eed)
    print(t)


def write_results(predictions, output_file='prediction.npy'):
    np.save(output_file, predictions)


def xavier_init(model):
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
        if hasattr(module, 'bias'):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
    return model


def load_my_state_dict(net, state_dict):
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
    return net


if __name__ == '__main__':
    import model
    import utils
    import net_sphere

    # all_cnn = train_net(layer_name='fc5_custom', pretrained_path='./model-big-resnet.pth', embedding_size=512, parts=[1], utterance_size=384, net=net_sphere.sphere20a, lr=0.000005, n_epochs=1, batch_size=1, num_workers=1)
    # all_cnn = train_net(layer_name='fc5_custom', embedding_size=100, net=model.all_cnn_module, lr=1e-5, n_epochs=500, batch_size=150, num_workers=4)

    # number_speakers = 381
    # sphere = net_sphere.sphere20a(number_speakers)
    # load_my_state_dict(sphere, torch.load('./model-big-resnet.pth'))
    # validate(net=sphere, layer_name='fc5_custom', batch_size=150, utterance_size=384, embedding_size=512, gpu=True,
    #          num_workers=6)
    # pred_similarities = infer_embeddings(net=sphere, layer_name='fc5_custom', utterance_size=384, embedding_size=512,
    #                                      gpu=True)

    tester = train_net(layer_name='lin1', pretrained_path=None, embedding_size=512, parts=[1], utterance_size=5184,
                       net=model.Tester, lr=0.005, n_epochs=350, batch_size=200, num_workers=6)
    pred_similarities = infer_embeddings(tester, layer_name='lin1', utterance_size=5184, embedding_size=512, gpu=True)
    write_results(pred_similarities.squeeze())
