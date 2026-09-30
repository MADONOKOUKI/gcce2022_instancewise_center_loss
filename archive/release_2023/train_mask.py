import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import hydra
import logging
import mlflow
from omegaconf import DictConfig, ListConfig

from dataset.cifar import *
from dataset.svhn import *
from dataset.stl10 import *
from models.WideResNet import *
from models.shakeshake import *
from models.resnet import *
from models.resnext import CifarResNeXt
from dataset.misc import progress_bar
from augmentation.cutout import *
from augmentation.autoaug import AutoAugment
from RandAugment import RandAugment
from augmentation.augmix import AugMix
from loss.center_loss import CenterLoss
from loss.contrastive_center_loss import ContrastiveCenterLoss
from models.densenet import *

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


# log = None
config = None
@hydra.main('config/train.yaml')
def main(cfg):
    log_params_from_omegaconf_dict(cfg)
    mlflow.set_experiment(cfg.mlflow.runname)
    solver = Solver(cfg)
    config = cfg
    solver.run()

log = logging.getLogger(__name__)
# fh = logging.FileHandler(str(cfg.data.name) +"-" + str(cfg.augmentation) +"-" + str(cfg.num_ensemble_imgs) )
# # ロガーに追加
# log.addHandler(fh)
# log = logging.getLogger(__name__)
    # log = logging.getLogger(str(cfg.data.name) +"-" + str(cfg.augmentation) +"-" + str(cfg.num_ensemble_imgs) )

# FileHandler の作成
# fh = logging.FileHandler('{:%Y-%m-%d}.log'.format(datetime.now()))
# # ロガーに追加
# log.addHandler(fh)

class Solver(object):
    def __init__(self, config):

        self.model_name = config.model.name
        self.model_num_classes = config.data.num_classes
        self.model = None
        self.lr = config.optim.lr
        self.epochs = config.optim.epochs
        self.train_batch_size = config.data.batch_size * 4
        self.alpha_rate = config.alpha_rate
        self.test_batch_size = config.data.batch_size * 4
        self.optim = config.optim
        self.cuda = True
        self.center_loss = None
        self.dataset = config.data.name
        self.augmentation = config.augmentation
        self.save_path = config.save_model_path
        self.regularization = torch.nn.MSELoss()
        self.num_imgs = config.num_ensemble_imgs
        self.config = config

    def load_data(self):

        if self.augmentation == "standard":
          train_transform = transforms.Compose([transforms.Resize(32), transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        if self.augmentation == "cutout":
          train_transform = transforms.Compose([transforms.Resize(32), transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor()])
          train_transform.transforms.append(Cutout(n_holes=self.config.cutout.n_holes, length=self.config.cutout.length))
        if self.augmentation == "randaug":
          train_transform = transforms.Compose([transforms.Resize(32), transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor()])
          train_transform.transforms.insert(0, RandAugment(self.config.randaug.N, self.config.randaug.M))
        if self.augmentation == "autoaug":
          train_transform = transforms.Compose([transforms.Resize(32), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), AutoAugment(),transforms.ToTensor()])
        if self.augmentation == "augmix":
          train_transform = transforms.Compose([transforms.Resize(32), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),AugMix(),transforms.ToTensor()])

        test_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])

        if self.dataset == "cifar10":
          train_set = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
          test_set = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        elif self.dataset == "cifar100":
          train_set = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
          test_set = CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        elif self.dataset == "svhn":
          train_set = SVHN(root='./data', split='train', download=True, transform=train_transform)
          test_set = SVHN(root='./data', split='test', download=True, transform=test_transform)
        elif self.dataset == "stl10":
          train_set = STL10(root='./data', split='train+unlabeled', download=True, transform=train_transform)
          test_set = STL10(root='./data', split='test', download=True, transform=test_transform)

        if self.config.regularization_loss_function == "L1":
            self.regularization = torch.nn.L1Loss()
        elif self.config.regularization_loss_function == "MSE":
            self.regularization = torch.nn.MSELoss()
        elif self.config.regularization_loss_function == "KL":
            self.regularization = torch.nn.KLDivLoss()
        elif self.config.regularization_loss_function == "Hubor":
            self.regularization = torch.nn.SmoothL1Loss()
          

        self.train_set = train_set
        if self.model_name == "resnext":
          self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size//2, shuffle=True)
          self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size//2, shuffle=False)

        else:
          self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
          self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.center_loss = ContrastiveCenterLoss(num_classes=self.model_num_classes, feat_dim=128, use_gpu=True)

        if self.model_name == "wideresnet":
          self.model = WideResNet(depth=self.config.model.depth, num_classes=self.model_num_classes, widen_factor=self.config.model.widen_factor, drop_rate=0.0).to(self.device)
        elif self.model_name == "shakeshake":
          self.model = ShakeShake(input_shape=(1, 3, 32, 32), n_classes=self.model_num_classes, base_channels=self.config.model.base_channels, depth=26).to(self.device)
        elif self.model_name == "resnet18":
          self.model = ResNet18(num_classes=self.model_num_classes).to(self.device)
        elif self.model_name == "densenet":
          self.model = DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=self.model_num_classes).to(self.device)
        elif self.model_name == "resnext":
          self.model = CifarResNeXt(num_classes=self.model_num_classes).to(self.device)

        if self.config.regularization == False and self.config.competitive_method == "center_loss":
          # self.center_loss = CenterLoss(num_classes=self.model_num_classes, feat_dim=342, use_gpu=True)
          # self.center_loss = CenterLoss(num_classes=self.model_num_classes, feat_dim=128, use_gpu=True)
          self.center_loss = CenterLoss(num_classes=self.model_num_classes, feat_dim=512, use_gpu=True)
          self.center_loss = CenterLoss(num_classes=self.model_num_classes, feat_dim=1024, use_gpu=True)
          # self.center_loss = CenterLoss(num_classes=self.model_num_classes, feat_dim=640, use_gpu=True)
          # self.center_loss = CenterLoss(num_classes=self.model_num_classes, feat_dim=128, use_gpu=True)
          self.optimizer = optim.SGD(list(self.model.parameters())+list(self.center_loss.parameters()), lr=self.optim.lr, momentum=self.optim.momentum, weight_decay=self.optim.weight_decay, nesterov=self.config.optim.nesterov)
        elif self.config.regularization == False and self.config.competitive_method == "contrastive_center_loss":
          # self.center_loss = ContrastiveCenterLoss(num_classes=self.model_num_classes, feat_dim=342, use_gpu=True) #densenet
          # self.center_loss = ContrastiveCenterLoss(num_classes=self.model_num_classes, feat_dim=640, use_gpu=True) #w28-10
          # self.center_loss = ContrastiveCenterLoss(num_classes=self.model_num_classes, feat_dim=128, use_gpu=True) #shakesgake
          self.center_loss = ContrastiveCenterLoss(num_classes=self.model_num_classes, feat_dim=512, use_gpu=True) # resnet-18
          self.center_loss = ContrastiveCenterLoss(num_classes=self.model_num_classes, feat_dim=1024, use_gpu=True) # resnet-18
          self.optimizer = optim.SGD(list(self.model.parameters())+list(self.center_loss.parameters()), lr=self.optim.lr, momentum=self.optim.momentum, weight_decay=self.optim.weight_decay, nesterov=self.config.optim.nesterov)
        else:  
          self.optimizer = optim.SGD(self.model.parameters(), lr=self.optim.lr, momentum=self.optim.momentum, weight_decay=self.optim.weight_decay, nesterov=self.config.optim.nesterov)
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.optim.scheduler.milestones, gamma=self.optim.scheduler.mul)
        # elif self.model_name == "shakeshake":
        #   self.model = ShakeShake(input_shape=(1, 3, 32, 32), n_classes=self.model_num_classes, base_channels=self.config.model.base_channels, depth=26).to(self.device)
        #   if self.config.regularization == False and self.config.competitive_method == "center_loss":
        #     self.center_loss = CenterLoss(num_classes=self.model_num_classes, feat_dim=2, use_gpu=True)
        #     self.optimizer = optim.SGD(list(self.model.parameters())+list(self.center_loss.parameters()), lr=self.optim.lr, momentum=self.optim.momentum, weight_decay=self.optim.weight_decay, nesterov=self.config.optim.nesterov)           
        #   elif self.config.regularization == False and self.config.competitive_method == "contrastive_center_loss":
        #     self.center_loss = ContrastiveCenterLoss(num_classes=self.model_num_classes, feat_dim=128, use_gpu=True)
        #     self.optimizer = optim.SGD(list(self.model.parameters())+list(self.center_loss.parameters()), lr=self.optim.lr, momentum=self.optim.momentum, weight_decay=self.optim.weight_decay, nesterov=self.config.optim.nesterov)
        #   else:
        #     self.optimizer = optim.SGD(self.model.parameters(), lr=self.optim.lr, momentum=self.optim.momentum, weight_decay=self.optim.weight_decay, nesterov=self.config.optim.nesterov)
          # T_max = len(self.train_set) * self.epochs
          # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max, eta_min=0, last_epoch=-1)

        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])

        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def train(self, epoch):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (imgs, index, target) in enumerate(self.train_loader):
            steps = epoch * len(self.train_loader) + batch_num
            self.optimizer.zero_grad()
            outputs = []
            features = []
            outputs_unlabeled = []
            loss, mean, mean_feature, mean_unlabeled = 0, 0, 0, 0
            means = torch.zeros(imgs[0].size()[0], self.model_num_classes).to(self.device)
            # means = torch.zeros(imgs[0].size()[0], 128).to(self.device)
            means_num = torch.zeros(imgs[0].size()[0]).to(self.device)
            for idx in range(self.num_imgs):
              if idx == 0 and self.config.competitive_method == "augmix":
                data, target = imgs[idx + 4].to(self.device), target.to(self.device)
              else:
                data, target = imgs[idx].to(self.device), target.to(self.device)
              output, feature = self.model(data)
              feature = feature.squeeze()
              outputs.append(output)
              # outputs_unlabeled.append(output_unlabeled)
              features.append(feature)

              # print(feature.size())
              mean += output
              mean_feature += feature
              # mean_feature += feature
              # mean += feature
              # for cls_idx in range(self.model_num_classes):
              #   # print(target)
              #   # print(cls_idx)
              #   # print((target == cls_idx).nonzero().squeeze())
              #   # print((target == cls_idx).sum())
              #   # exit()
              #   means[(target == cls_idx).nonzero().squeeze()] += output[(target == cls_idx).nonzero().squeeze()].mean()
              #   # means[(target == cls_idx).nonzero().squeeze()] += feature[(target == cls_idx).nonzero().squeeze()].sum()
              #   # means_num[(target == cls_idx).nonzero().squeeze()] += (target == cls_idx).sum()

            mean /= self.num_imgs
            mean_feature /= self.num_imgs
            # if seld.dataset == "stl10":
            #   mean_unlabeled /= self.num_imgs
            # print(means.size(), means_num.size())
            # print(means)
            # print(means_num)
            # means /= means_num.unsqueeze(-1)
            # print(means)
            # exit()
            # for cls_idx in range(self.model_num_classes):
            #     # means[(target == cls_idx).nonzero().squeeze()] = (mean[(target == cls_idx).nonzero().squeeze()] + (means[(target == cls_idx).nonzero().squeeze()] / len((target == cls_idx))).squeeze() )/2.0
                
            #     print(means[(target == cls_idx).nonzero().squeeze()])


            #     means[(target == cls_idx).nonzero().squeeze()] = (mean[(target == cls_idx).nonzero().squeeze()] + (means[(target == cls_idx).nonzero().squeeze()] ).squeeze() )/ ((target == cls_idx).sum() + 1)
                
            #     print(means[(target == cls_idx).nonzero().squeeze()])
                # print(means[(target == cls_idx).nonzero().squeeze()].size(), means_num[(target == cls_idx).nonzero().squeeze()].size())
            #     if means_num[(target == cls_idx).nonzero().squeeze()] is None:
            #       continue
            #     print(means_num[(target == cls_idx).nonzero().squeeze()])
            #     print(means_num[(target == cls_idx).nonzero().squeeze()][0])
            #     means[(target == cls_idx).nonzero().squeeze()] = means[(target == cls_idx).nonzero().squeeze()] / (means_num[(target == cls_idx).nonzero().squeeze()])[0]

            # loss += 0.5 * (self.regularization(outputs[0], outputs[1].detach()) +  self.regularization(outputs[1], outputs[0].detach()))
            for idx in range(self.num_imgs):
                if self.config.regularization == True:
                  if self.config.regularization_loss_function == "KL":
                    loss += self.criterion(outputs[idx], target) + self.regularization(F.softmax(outputs[idx]).log(), F.softmax(mean))
                  else:
                    mask = (target  != -1)
                    no_mask = (target  == -1)
                    if mask.sum() > 0:
                      triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
                      # loss += self.alpha_rate * (self.regularization(outputs[idx], mean.detach()) ) +  (1 - self.alpha_rate) * self.criterion(outputs[idx][mask], target[mask]) * (len(mask) / mask.sum() )#+ self.regularization(self.model.fc(mean_feature), mean.detach()) #+ self.center_loss(features[idx], target) * alpha
                      # loss += self.alpha_rate * (self.regularization(outputs[idx], mean.detach()) )  +  (1 - self.alpha_rate) * self.criterion(outputs[idx][mask], target[mask])  #+ # * (len(mask) / mask.sum() )#+ self.regularization(self.model.fc(mean_feature), mean.detach()) #+ self.center_loss(features[idx], target) * alpha
                      # loss += self.alpha_rate * (self.regularization(outputs[idx], mean.detach()) )  +  (1 - self.alpha_rate) * self.criterion(outputs[idx], target) 
                      loss += self.alpha_rate * (self.regularization(outputs[idx], mean.detach()) )  +  (1 - self.alpha_rate) * self.criterion(outputs[idx], target) 
                      # loss += self.alpha_rate * (self.regularization(outputs[idx], mean.detach()) )  +   self.criterion(outputs[idx], target) 
                      # loss += self.alpha_rate * (self.regularization(outputs[idx], mean) )  +   self.criterion(outputs[idx], target) 
                      # loss += self.alpha_rate * (self.regularization(outputs[idx], mean.detach()) )  +  (1 - self.alpha_rate) * self.criterion(outputs[idx], target) 
                      # loss += self.alpha_rate * (self.regularization(feature[idx], mean_feature.detach()) )  +  (1 - self.alpha_rate) * self.criterion(outputs[idx], target) 
                      # pos = outputs[idx].size()[0] - 1
                      # loss += 0.5 * triplet_loss(mean, outputs[idx], torch.cat([outputs[(idx+1)%self.num_imgs][pos-1:], outputs[(idx+1)%self.num_imgs][:pos-1] ], axis=-0)) #+ # * (len(mask) / mask.sum() )#+ self.regularization(self.model.fc(mean_feature), mean.detach()) #+ self.center_loss(features[idx], target) * alpha
                      # kl =  torch.nn.KLDivLoss()
                      # loss += 0.5 * kl(F.softmax(outputs_unlabeled[idx][no_mask]).log(), F.softmax(mean[no_mask]))
                      # print(kl(F.softmax(outputs_unlabeled[idx][no_mask]).log(), F.softmax(outputs[idx][no_mask])))
                      # exit()
                      # loss += self.alpha_rate * (self.regularization(outputs[idx], mean.detach()) ) +  (1 - self.alpha_rate) * self.criterion(outputs[idx][mask], target[mask])#+ self.regularization(self.model.fc(mean_feature), mean.detach()) #+ self.center_loss(features[idx], target) * alpha
                    else:
                      loss += self.alpha_rate * (self.regularization(outputs[idx], mean.detach()) ) #+ self.regularization(self.model.fc(mean_feature), mean.detach()) #+ self.center_loss(features[idx], target) * alpha
                    # loss += self.regularization(output[idx], (mean + means ) /2)  + self.criterion(outputs[idx], target)
                elif self.config.regularization == False and (self.config.competitive_method == "center_loss" or self.config.competitive_method == "contrastive_center_loss"):
                    alpha = 0.1
                    loss += self.criterion(outputs[idx], target) + self.center_loss(features[idx], target) * alpha
                elif self.config.regularization == False and self.config.competitive_method == "triplet_loss":
                  triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
                  pos = outputs[idx].size()[0] - 1
                  # print(outputs[(idx+1)%self.num_imgs][pos].size())
                  # print(outputs[(idx+1)%self.num_imgs][:pos].size())
                  loss += self.criterion(outputs[idx], target) + 0.1 * triplet_loss(outputs[idx], outputs[(idx+1)%self.num_imgs], torch.cat([outputs[(idx+1)%self.num_imgs][pos-1:], outputs[(idx+1)%self.num_imgs][:pos-1] ], axis=-0))
                elif self.config.regularization == False and self.config.competitive_method == "augmix":

                  p_clean, p_aug1, p_aug2 = F.softmax(outputs[0], dim=1), F.softmax(outputs[1], dim=1), F.softmax(outputs[2], dim=1)

                  # Clamp mixture distribution to avoid exploding KL divergence
                  p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                  loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

                  loss += self.criterion(outputs[0], target) 
                else:
                    loss += self.criterion(outputs[idx], target) 
            loss /= self.num_imgs
            # mlflow.log_metric("train loss", loss.item(), step=steps)

            loss.backward()

            # exit()
            # if self.config.regularization == True:
            #     for param in self.center_loss.parameters():
            #         alpha = 0.1
            #         param.grad.data *= (1./alpha)

            if self.config.regularization == False and (self.config.competitive_method == "center_loss" or self.config.competitive_method == "contrastive_center_loss"):
                for param in self.center_loss.parameters():
                    alpha = 0.1
                    param.grad.data *= (1./alpha)

            mask = (target  != -1)
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target[mask].size(0)
            train_correct += np.sum(prediction[1][mask].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))
        return train_loss, train_correct / total

    def test(self, epoch, test_loader=None):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        if test_loader != None:
          loader = test_loader
        else:
          loader = self.test_loader

        with torch.no_grad():
            for batch_num, (data, _, target) in enumerate(loader):
                steps = epoch * len(loader) + batch_num
                data, target = data[0].to(self.device), target.to(self.device)
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                mlflow.log_metric("test loss", loss.item(), step=steps)

                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def test_c(self, epoch):

      """Evaluate network on given corrupted dataset."""
      corruption_accs = []
      base_path = "/fs1/groups1/gaa50073/madono/bmvc2021/em_classifier/data/CIFAR-10-C/"
      CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
      ]

      test_data = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]))

      for corruption in CORRUPTIONS:
        
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
          test_data,
          batch_size=self.train_batch_size,
          shuffle=False)

        test_loss, test_acc = self.test(epoch, test_loader)
        corruption_accs.append(test_acc)
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(corruption, test_loss, 100 - 100. * test_acc))

      return np.mean(corruption_accs)

    def save(self):
        model_out_path = self.save_path
        torch.save(self.model.state_dict(), 'model_tmp.pth')
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        # mdl = 
        # torch.save(self.model.to('cpu').state_dict(), 'model_tmp.pth')
        # self.model.to('gpu')
    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            train_result = self.train(epoch)
            test_result = self.test(epoch)
            # if epoch % 20 ==0:
            #   test_result2 = self.test_c(epoch)
            if accuracy <= test_result[1]:
              self.save()
              accuracy = test_result[1]

            # logging at the hydra
            mlflow.log_metric("train acc", train_result[1] * 100, step=epoch)
            mlflow.log_metric("test acc", test_result[1] * 100, step=epoch)
            log.info("Epoch {} : train loss {:.3f}, train acc. {:.3f}".format(epoch, train_result[0],  train_result[1] * 100))
            log.info("Epoch {} : test loss {:.3f}, test acc. {:.3f}".format(epoch, test_result[0],  test_result[1] * 100))
            # if epoch % 20 ==0:
            #   log.info("Epoch {} : test c acc. {:.3f}".format(epoch, test_result2 * 100))
        # print(accuracy)
        log.info("Best acc. {:.3f}", accuracy)


if __name__ == '__main__':
    main()

