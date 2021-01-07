import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
# custom module
from dataset import TrainDataset
from model import SRFBN
from utils import IOStream, AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 5"

batch_size = 18
epochs = 250
io = IOStream('run.log')
trainingimagesize = 144
augmented_data_path = 'training_data'

def prepare_data():
    ''' prepare training data
        for each image, keep a copy in the train dataset
    even if it's smaller than crop size (but we have to
    resize it), and for larger image, generate smaller
    image which is its copy.
    '''
    os.system("rm -r " + augmented_data_path)
    os.makedirs(augmented_data_path, exist_ok=True)
    for filename in glob.glob('training_hr_images/*.png'):
        img = cv2.imread(filename)
        H, W, _ = img.shape

        # resize small images
        if H < trainingimagesize or W < trainingimagesize:
            scale = max(trainingimagesize/H,
                        trainingimagesize/W)
            H = int(H * scale + 1)
            W = int(W * scale + 1)
            img = cv2.resize(img, (W, H))

        # write image
        new_filename = filename.replace("training_hr_images",
                                        augmented_data_path)
        cv2.imwrite(new_filename, img)
        cv2.imwrite(new_filename, img[:, ::-1])

        # for large image, generate augmented images
        for scale in [0.9, 0.8, 0.7, 0.6, 0.5]:
            _H, _W = int(H * scale) + 1, int(W * scale) + 1
            if _H < trainingimagesize or _W < trainingimagesize:
                break
            new_img = cv2.resize(img, (_W, _H))
            cv2.imwrite(new_filename.replace(".", "_%.1f." % scale), new_img)
            cv2.imwrite(new_filename.replace(".", "_%.1f." % scale),
                        new_img[:, ::-1])


def LoadData():
    ''' Load Training Data'''

    # prepare data
    prepare_data()

    # get image indices of train/test dataset
    filenames = np.array(glob.glob('%s/*.png' % augmented_data_path))
    L = len(filenames)
    perm = np.random.permutation(L)
    TrainIndices = perm[L//10:]
    TestIndices = perm[:L//10]

    # create datasets
    training_set = TrainDataset(filenames[TrainIndices],
                                trainingimagesize, 'train')
    test_set = TrainDataset(filenames[TestIndices],
                            trainingimagesize, 'val')

    # create dataloaders
    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "num_workers": 32}
    test_params = {"batch_size": 1,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": 32}
    train_loader = DataLoader(training_set, **training_params)
    test_loader = DataLoader(test_set, **test_params)

    return train_loader, test_loader


def cal_PSNR(HR, pred_HR):
    # calculate PSNR
    MSE = (HR - pred_HR) ** 2
    MSE = torch.mean(MSE, 3)
    MSE = torch.mean(MSE, 2)
    MSE = torch.mean(MSE, 1)
    psnr = 10 * torch.log10(255. * 255. / MSE)
    return torch.mean(psnr, 0)


def main():
    ''' main function '''
    torch.cuda.manual_seed(1)

    # load data
    train_loader, test_loader = LoadData()

    # define model
    model = SRFBN(in_channels=3,
                  out_channels=3,
                  num_features=64,
                  num_steps=4,
                  num_groups=6,
                  act_type='prelu',
                  norm_type=None)
    model = model.cuda()
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load('checkpoint.t7'))

    # define optimizer and schduler
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4, amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer, epochs)

    max_psnr = 0
    # start training
    for epoch in range(epochs):
        io.cprint('Epoch[%d]' % epoch)

        # train
        train(model, train_loader, optimizer)
        torch.save(model.state_dict(), 'checkpoint.t7')

        # do validation
        psnr = val(model, test_loader)
        if psnr > max_psnr:
            max_psnr = psnr
            torch.save(model.state_dict(), 'SRFBNx3.t7')

        scheduler.step()


def train(model, train_loader, optimizer):
    ''' Training '''
    model.train()

    # Object for visualizing
    Loss = AverageMeter()

    criterion = nn.L1Loss()
    for i, (HR, LR) in enumerate(train_loader):
        HR = HR.cuda().float().permute(0, 3, 1, 2)
        LR = LR.cuda().float().permute(0, 3, 1, 2)
        bs, _, _, _ = HR.shape

        # predict and do back propagation
        pred_HRs = model(LR)
        loss = 0
        for pred_HR in pred_HRs:
            loss = loss + criterion(HR, pred_HR)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record and visualize losses
        Loss.update(float(loss), bs)
        print('[{0}/{1}]  '
              'Loss {Loss.val:.7f} ({Loss.avg:.7f})     '
              .format(i + 1, len(train_loader),
                      Loss=Loss), end='\r')

    # save log
    print(' ' * 100, end='\r')
    io.cprint(' * Train * Loss {Loss.avg:.7f}  '
              .format(Loss=Loss))


def val(model, test_loader):
    ''' Validation '''
    model.eval()
    criterion = cal_PSNR

    PSNR = AverageMeter()

    with torch.no_grad():
        for i, (HR, LR) in enumerate(test_loader):
            # load data
            HR = HR.cuda().float().permute(0, 3, 1, 2)
            LR = LR.cuda().float().permute(0, 3, 1, 2)
            bs, _, H, W = HR.shape

            # calculate psnr
            pred_HRs = model(LR)
            pred_HR = pred_HRs[-1]
            psnr = criterion(HR, pred_HR)

            # record and visualize psnr
            PSNR.update(psnr, bs)
            print('[{0}/{1}]  '
                  'PSNR (last) {PSNR.val:.7f} ({PSNR.avg:.7f})     '
                  .format(i + 1, len(test_loader),
                          PSNR=PSNR), end='\r')

        # save log
        print(' ' * 100, end='\r')
        io.cprint(' * Test * PSNR {PSNR.avg:.7f}  '
                  .format(PSNR=PSNR))
    return PSNR.avg


if __name__ == "__main__":
    main()
