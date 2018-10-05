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

    def download(self):
        if not os.path.exists('./data'):
            os.makedirs('./data')
        print('Downloading tar files...')
        p = subprocess.Popen('wget https://11785fall2018.s3.amazonaws.com/hw2p2_A.tar.gz /data/hw2p2_A.tar.gz',
                             shell=True)
        p.wait()
        p = subprocess.Popen('wget https://11785fall2018.s3.amazonaws.com/hw2p2_B.tar.gz /data/hw2p2_B.tar.gz',
                             shell=True)
        p.wait()
        p = subprocess.Popen('wget https://11785fall2018.s3.amazonaws.com/hw2p2_C.tar.gz /data/hw2p2_C.tar.gz',
                             shell=True)
        p.wait()
        print('Downloaded files to ./data.')

    def extract(self):
        print('Extracting tar files...')
        p = subprocess.Popen('tar -xvzf /data/hw2p2_A.tar.gz --strip 1 -C data', shell=True)
        p.wait()
        os.remove('./data/hw2p2_A.tar.gz')
        p = subprocess.Popen('tar -xvzf /data/hw2p2_B.tar.gz --strip 1 -C data', shell=True)
        p.wait()
        os.remove('./data/hw2p2_B.tar.gz')
        p = subprocess.Popen('tar -xvzf /data/hw2p2_C.tar.gz --strip 1 -C data', shell=True)
        p.wait()
        os.remove('./data/hw2p2_C.tar.gz')
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
    import sys
    if len(sys.argv) < 3 or sys.argv[2] not in list(map(str, range(1, 7))) + ["dev", "test"]:
        print("Usage:", sys.argv[0], "<path to npz files>", "<chunk among {1, 2, .., 6, dev, test}>")
        exit(0)

    # path, part = sys.argv[1], sys.argv[2]
    print('Setting to {} frames.'.format(sys.args[1]))
    dl = DataDownload(vad_nframes=int(sys.args[1]))
    dl.download()
    dl.get_train()
    dl.get_dev()
    dl.get_test()
    print('Download and pre-processing successful.')
    # print(UtteranceTrainDataset()[1][1])
    # UtteranceTestDataset()[1]
    # print(UtteranceValidationDataset()[1][2])
