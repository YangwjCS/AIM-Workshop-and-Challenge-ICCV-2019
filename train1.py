import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from dataset.data import myDataloader
import scipy.misc
import imageio
from network.loss import *
from network.SSIM import SSIM
import random
import re

from network.Res29_1 import Ensemble

parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=2, help="Training step")
parser.add_argument("--nEpochs", type=int, default=60, help="Number of epochs to train")
parser.add_argument("--lrG", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument("--lrD", type=float, default=1e-3, help="Learning rate, default=1e-4")
parser.add_argument("--step", type=int, default=20, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.1, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--scale", default=4, type=int, help="Scale factor, Default: 4")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--gated", type=bool, default=True, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#def mkdir_steptraing():
#    root_folder = os.path.abspath('.')
#    models_folder = join(root_folder, 'models/modelW')
#    step1_folder, step2_folder, step3_folder = join(models_folder,'1'), join(models_folder,'2'), join(models_folder, '3')
#    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder) and os.path.exists(step3_folder)
#    if not isexists:
#        os.makedirs(step1_folder)
#        os.makedirs(step2_folder)
#        os.makedirs(step3_folder)
#        print("===> Step training models store in models/1 & /2 & /3.")


def mkdir_model(path):
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, path)
    isexists = os.path.exists(models_folder)
    if not isexists:
        os.makedirs(models_folder)

        print("===> Step training models store in models/1 & /2 & /3.")


def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


#def which_trainingstep_epoch(resume):
#    trainingstep = "".join(re.findall(r"\d", resume)[0])
#    start_epoch = "".join(re.findall(r"\d", resume)[1:])
#    return int(trainingstep), int(start_epoch)


class trainer_S2_2:
    def __init__(self, train_gen, step, numD=4):
        super(trainer_S2_2, self).__init__()

        self.numd = numD
        self.step = step
        self.trainloader = train_gen
        self.modelG = Ensemble().cuda()
        #self.modelG.load_state_dict(torch.load("/home/ywj/game/models/modelOut/MS29/Dem_44.pkl"))
        print("#############################")
        #print_network(self.modelG)
        print("#############################")

        self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.modelG.parameters()), lr=1e-4, betas=(0.5, 0.9))

        criterion =nn.L1Loss()
        self.SSIMLoss = SSIM().cuda()
        self.criterion = criterion.cuda()

    def opt_G(self, fake, real):
        self.optimizer_G.zero_grad()
        g_loss_MSE = self.criterion(fake, real.detach())

        l_SSIM = 1 - self.SSIMLoss(fake, real).mean()

        g_loss = g_loss_MSE*0.75 + l_SSIM*1.1
        g_loss.backward()

        self.optimizer_G.step()

        return g_loss_MSE, l_SSIM

    def adjust_learning_rate(self, epoch):
        lrG = opt.lrG * (opt.lr_decay ** (epoch // opt.step))
        print(lrG)
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lrG

    def checkpoint(self, epoch):
        path = "models/modelOut/MS{}".format(58)
        mkdir_model(path)
        model_out_path =path + "/Dem_{}.pkl".format(epoch)
        torch.save(self.modelG.state_dict(), model_out_path)
        print("===>Checkpoint saved to {}".format(model_out_path))

    def train(self, epoch):
        self.checkpoint(epoch)

        self.adjust_learning_rate(epoch - 1)
        epoch_loss = 0
        ssim_loss = 0
        for iteration, (Ix, Jx) in enumerate(self.trainloader):
            Ix = Ix.to(device)
            Jx = Jx.to(device)

            fake = self.modelG(Ix)
            g_loss_MSE, lossSSIM = self.opt_G(fake, Jx)

            epoch_loss += g_loss_MSE.cpu().data
            ssim_loss += lossSSIM.cpu().data

            if iteration % 100 == 0:
                print(
                    "===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch, iteration, len(trainloader), g_loss_MSE.cpu()))

                Ix_cc = fake  # modelD(Detail_I) #+ Ix[:, 6:9, :, :] modelD(Detail_I)#
                Ix_cc = Ix_cc.clamp(0, 1)
                Ix_cc = Ix_cc[0].permute(1, 2, 0).detach().cpu().numpy()

                Ix = Ix.clamp(0, 1)
                Ix = Ix[0].permute(1, 2, 0).detach().cpu().numpy()
                Ix_cc = np.hstack([Ix, Ix_cc])

                Jx = Jx.clamp(0, 1)
                Jx = Jx[0].permute(1, 2, 0).detach().cpu().numpy()
                Ix_cc = np.hstack([Ix_cc, Jx])

                # print(Ix_cc.shape)
                imageio.imwrite('./results' + '/' + str((epoch - 1) * 100 + iteration / 100) + '.png', np.uint8(Ix_cc*255))
                #print("MSE:" + str(g_loss_MSE.cpu().data) + 'SSIM:' + str(lossSSIM.cpu().data))
                print("MSE:{:4f},SSIM:{:4f}".format(g_loss_MSE, lossSSIM))
        print("===>Epoch{} Complete: Avg loss is :L1:{:4f},SSIM:{:4f} ".format(epoch, epoch_loss / len(trainloader), ssim_loss/ len(trainloader)))

trainloader, testloader = myDataloader().getLoader()
opt = parser.parse_args()
opt.seed = random.randint(1, 10000)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)


for i in range(1, 2):
    print("===> Loading model and criterion")

    #trainModel = trainer_S1(trainloader, step=i, numD=1)
    trainModel = trainer_S2_2(trainloader, step=1, numD=1)

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Step {}:-------------------------------".format(i))
        trainModel.train(epoch)
