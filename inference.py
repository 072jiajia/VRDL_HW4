import os
import cv2
import glob
import torch
import torch.nn as nn
# custom module
from model import SRFBN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # create output directory
    output_path = 'prediction'
    os.makedirs(output_path, exist_ok=True)

    # load model
    model_path = 'SRFBNx3.t7'
    model = SRFBN(3, 3, 64, 4, 6).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for filename in glob.glob('testing_lr_images/*'):
            # read image
            LR = cv2.imread(filename)
            print("image:", filename)
            print("shape:", LR.shape)
            LR = torch.tensor([LR]).cuda().float().permute(0, 3, 1, 2)

            # inference
            pred_HR = model(LR)[-1]
            pred_HR = pred_HR.permute(0, 2, 3, 1).cpu().detach().numpy()[0]

            # write image
            newfilename = filename.replace('testing_lr_images', output_path)
            cv2.imwrite(newfilename, pred_HR)
