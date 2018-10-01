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
import utils
import os


class DataLoader:
    def __init__(self):
        self.files = {1: 'hw2p2_A.tar.gz', 2: 'hw2p2_B.tar.gz', 3: 'hw2p2_C.tar.gz'}

    def getdev(self):
        """
        Given path to the dev.preprocessed.npz file, loads and returns:
        (1) Dev trials list, where each item is [enrollment_utterance_idx, test_utterance_idx]
        (2) Dev trials labels, where each item is True if same speaker (and False otherwise)
        (3) (Dev) Enrollment array of utterances
        (4) (Dev) Test array of utterances
        :return: [enrollment_utterance_idx, test_utterance_idx], labels, enrollment, test
        """
        try:
            f = utils.dev_load('./data/1.npz')
            return f
        except Exception:
            if not os.path.exists('./data'):
                os.makedirs('./data')
            p = subprocess.Popen('wget https://11785fall2018.s3.amazonaws.com/hw2p2_A.tar.gz /data/hw2p2_A.tar.gz',
                                 shell=True)
            p.wait()
            p = subprocess.Popen('tar -xvzf ./data/hw2p2_A.tar.gz --strip 1 -C data', shell=True)
            p.wait()
            # os.remove('./data/hw2p2_A.tar.gz')


if __name__ == "__main__":
    dl = DataLoader()
    dl.getdev()