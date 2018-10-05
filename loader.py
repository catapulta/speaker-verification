# import boto3
# import botocore
#

#
# bucket = '11785fall2018'
#
# s3 = boto3.resource('s3')
#
# for file in files[:1]:
#     try:
#         s3.Bucket(bucket).download_file(file, os.path.basename(file))
#     except botocore.exceptions.ClientError as e:
#         if e.response['Error']['Code'] == "404":
#             print("The object does not exist.")
#         else:
#             raise

import subprocess
import os
from torch.utils.data import Dataset
import numpy as np
import utils
import torch
import preprocess_plus


class DataDownload:
    def __init__(self, vad_nframes):
        self.vad_nframes = vad_nframes

    def download(self, parts=['A', 'B']):
        if not os.path.exists('./data'):
            os.makedirs('./data')
        print('Downloading tar files...')
        for file in parts:
            p = subprocess.Popen(
                'wget -O /data/hw2p2_{0}.tar.gz https://11785fall2018.s3.amazonaws.com/hw2p2_{0}.tar.gz'.format(file),
                shell=True)
            p.wait()
        print('Downloaded files to ./data.')

    def extract(self, parts=['A', 'B'], erase_tar=False):
        print('Extracting tar files...')
        for file in parts:
            p = subprocess.Popen('tar -xvzf /data/hw2p2_{}.tar.gz --strip 1'.format(file), shell=True)
            p.wait()
            if erase_tar:
                os.remove('./data/hw2p2_{}.tar.gz'.format(file))
        print('Extracted files to ./data.')

    def get_train(self, parts=[1]):
        for i in parts:
            input_path = './data/{}.npz'.format(i)
            output_path = './data/{}.preprocessed.npz'.format(i)
            print('Pre-processing file {}...'.format(i))
            npz = np.load(input_path, encoding='latin1')
            np.savez(output_path, feats=preprocess_plus.bulk_VAD(npz['feats'], self.vad_nframes),
                     targets=npz['targets'])

    def get_dev(self):
        input_path = './data/dev.npz'
        output_path = './data/dev.preprocessed.npz'
        print('Pre-processing dev file...')
        npz = np.load(input_path, encoding='latin1')
        np.savez(output_path, enrol=preprocess_plus.bulk_VAD(npz['enrol'], self.vad_nframes),
                 test=preprocess_plus.bulk_VAD(npz['test'], self.vad_nframes), trials=npz['trials'],
                 labels=npz['labels'])

    def get_test(self):
        input_path = './data/test.npz'
        output_path = './data/test.preprocessed.npz'
        print('Pre-processing test file...')
        npz = np.load(input_path, encoding='latin1')
        np.savez(output_path, enrol=preprocess_plus.bulk_VAD(npz['enrol'], self.vad_nframes),
                 test=preprocess_plus.bulk_VAD(npz['test'], self.vad_nframes), trials=npz['trials'])


class UtteranceTrainDataset(Dataset):
    def __init__(self, path='./data', parts=[1], transform=False):
        self.features, self.labels, self.n_labels = utils.train_load(path, parts)
        self.num_entries = len(self.features)

    def __getitem__(self, index):
        features = torch.from_numpy(self.features[index]).float()
        features = features.view(-1, 1, features.shape[1], features.shape[2])
        labels = torch.from_numpy(np.array(self.labels[index])).long()
        return labels, features

    def __len__(self):
        return self.num_entries


class UtteranceValidationDataset(Dataset):
    def __init__(self, path='./data/dev.preprocessed.npz'):
        self.trials, self.labels, self.enrol, self.test = utils.dev_load(path)
        self.num_entries = len(self.labels)

    def __getitem__(self, index):
        trial = self.trials[index]
        labels = torch.from_numpy(np.array(self.labels[index]).astype(int)).long()
        enrol = torch.from_numpy(self.enrol[trial[0]]).float()
        test = torch.from_numpy(self.test[trial[1]]).float()
        return labels, enrol, test

    def __len__(self):
        return self.num_entries


class UtteranceTestDataset(Dataset):
    def __init__(self, path='./data'):
        self.trials, self.enrol, self.test = utils.test_load(path)
        self.num_entries = len(self.trials)

    def __getitem__(self, index):
        trial = self.trials[index]
        enrol = torch.from_numpy(self.enrol[trial[0]]).float()
        test = torch.from_numpy(self.test[trial[1]]).float()
        return enrol, test

    def __len__(self):
        return self.num_entries


if __name__ == "__main__":
    dl = DataDownload(vad_nframes=10000)
    dl.download(parts=['A', 'B'])
    dl.extract(parts=['A', 'B'], erase_tar=False)
    dl.get_train(parts=[1, 2])
    dl.get_dev()
    print('Download and pre-processing successful.')
    # print(UtteranceTrainDataset()[1][1])
    # UtteranceTestDataset()[1]
    # print(UtteranceValidationDataset()[1][2])
