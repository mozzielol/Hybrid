import struct
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch


def read_seq(path):
    def read_header(ifile):
        feed = ifile.read(4)
        norpix = ifile.read(24)
        version = struct.unpack('@i', ifile.read(4))
        length = struct.unpack('@i', ifile.read(4))
        assert (length != 1024)
        descr = ifile.read(512)
        params = [struct.unpack('@i', ifile.read(4))[0] for i in range(0, 9)]
        fps = struct.unpack('@d', ifile.read(8))
        # skipping the rest
        ifile.read(432)
        image_ext = {100: 'raw', 102: 'jpg', 201: 'jpg', 1: 'png', 2: 'png'}
        return {'w': params[0], 'h': params[1],
                'bdepth': params[2],
                'ext': image_ext[params[5]],
                'format': params[5],
                'size': params[4],
                'true_size': params[8],
                'num_frames': params[6]}

    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    # this is freaking magic, but it works
    extra = 8
    s = 1024
    seek = [0] * (params['num_frames'] + 1)
    seek[0] = 1024

    images = []

    for i in range(0, params['num_frames'] - 1):
        tmp = struct.unpack_from('@I', bytes[s:s + 4])[0]
        s = seek[i] + tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s + 1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        seek[i + 1] = s
        nbytes = struct.unpack_from('@i', bytes[s:s + 4])[0]
        I = bytes[s + 4:s + nbytes]

        tmp_file = '/tmp/img%d.jpg' % i
        open(tmp_file, 'wb+').write(I)

        img = cv2.imread(tmp_file)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images


class Sequence:
    def __init__(self, duration, interval, directory='./data/sports'):
        self.duration = duration
        self.interval = interval
        self.transform = Transform()
        self.data = self._load_data(directory)
        self.counter = 0

    def _load_data(self, directory):
        data = {}
        target = []
        for sample, cls in self._iter_dir(directory):
            try:
                data[cls].append(sample)
            except KeyError:
                data[cls] = []
                data[cls].append(sample)
            target.append(cls)

        for idx, t in enumerate(np.unique(np.array(target))):
            data[idx] = data[t]
            del data[t]
        return data

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter > len(self):
            self.counter = 0
            raise StopIteration
        cls_id = np.random.randint(len(self.data.keys()))
        sample_id = np.random.randint(len(self.data[cls_id]))
        data = self.transform(read_seq(self.data[cls_id][sample_id]))
        frame_id = np.random.randint(self.duration * self.interval,
                                     len(data) - self.duration * self.interval)
        self.counter += 1
        return (data[frame_id - self.duration * self.interval: frame_id: self.interval],
                data[frame_id].unsqueeze(0),
                data[frame_id: frame_id + self.duration * self.interval: self.interval]), torch.Tensor(cls_id).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _iter_dir(directory):
        for dir in os.listdir(directory):
            sample_counter = 0
            if os.path.isdir(os.path.join(directory, dir)):
                for filename in os.listdir(os.path.join(directory, dir)):
                    if filename.endswith(".seq") and sample_counter < 20:
                        sample_counter += 1
                        yield os.path.join(directory, dir, filename), dir
                    else:
                        continue

    def train_test_split(self, train_sample_per_class=2, test_sample_per_class=20, replace=True):
        """
        :param train_sample_per_class: number of labelled sample per class
        :param test_sample_per_class: number of test sample per class
        :return:
        """
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for cls_id in self.data.keys():
            train_sample_idx = []
            for _ in range(train_sample_per_class):
                sample_id = np.random.randint(len(self.data[cls_id]))
                if not replace:
                    while sample_id in train_sample_idx or np.random.random() > 0.5:
                        sample_id = np.random.randint(len(self.data[cls_id]))
                    train_sample_idx.append(sample_id)
                data = self.data[cls_id][sample_id]
                frame_id = np.random.randint(len(data))
                train_x.append(data[frame_id])
                train_y.append(cls_id)

            for _ in range(test_sample_per_class):
                sample_id = np.random.randint(len(self.data[cls_id]))
                if not replace:
                    while sample_id in train_sample_idx:
                        sample_id = np.random.randint(len(self.data[cls_id]))
                data = self.data[cls_id][sample_id]
                frame_id = np.random.randint(len(data))
                test_x.append(data[frame_id])
                test_y.append(cls_id)
                train_sample_idx.append(sample_id)

        return np.stack(train_x), np.array(train_y), np.stack(test_x), np.array(test_y)


class Transform(object):
    def __init__(self):
        self.tensor_transformer = transforms.Compose([
            transforms.ToTensor()])

    def __call__(self, sample):
        sample = np.array(sample)
        try:
            sample = sample.transpose(0, 3, 1, 2)
        except ValueError:
            pass
        return torch.Tensor(sample.astype(np.float32))
