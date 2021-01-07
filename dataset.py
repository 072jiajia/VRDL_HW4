import cv2
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    ''' TrainDataset
     - partition:  training or validating
     - crop_size:  crop size of training images
    '''

    def __init__(self, filenames, crop_size, partition):
        if partition == "train":
            assert crop_size % 3 == 0
            self.crop_size = crop_size
        self.image = list(filenames)
        self.partition = partition

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        if type(self.image[item]) == np.str_:
            self.image[item] = cv2.imread(str(self.image[item]))
        H, W, _ = self.image[item].shape

        # get corresponding data
        if self.partition == 'train':
            # get croping top, left, bottom, right
            top = np.random.randint(0, H - self.crop_size + 1)
            left = np.random.randint(0, W - self.crop_size + 1)

            # Crop and do augmentation
            HR = self.image[item][top: top + self.crop_size,
                                  left: left + self.crop_size]

            # create LR image
            LR = cv2.resize(HR, (self.crop_size//3, self.crop_size//3))
        else:
            cropH = H//3 * 3
            cropW = W//3 * 3
            HR = self.image[item][:cropH, :cropW]
            LR = cv2.resize(HR, (W//3, H//3))

        return HR, LR
