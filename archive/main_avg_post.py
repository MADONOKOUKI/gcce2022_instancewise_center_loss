import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
from cifar import *
import argparse

from models.WideResNet import *
from misc import progress_bar
import torch.nn.functional as F

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=64, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=64, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--num_imgs', default=2, type=int, help='num of augmented images')

    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


def to_one_hot_vector(num_class, label):
    b = np.zeros((label.shape[0], num_class))
    b[np.arange(label.shape[0]), label] = 1

    return b

class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.num_imgs = config.num_imgs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # self.model = LeNet().to(self.device)
        # self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
        # self.model = VGG16().to(self.device)
        # self.model = VGG19().to(self.device)
        # self.model = GoogLeNet().to(self.device)
        # self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        # self.model = resnet50().to(self.device)
        # self.model = resnet101().to(self.device)
        # self.model = resnet152().to(self.device)
        # self.model = DenseNet121().to(self.device)
        # self.model = DenseNet161().to(self.device)
        # self.model = DenseNet169().to(self.device)
        # self.model = DenseNet201().to(self.device)
        self.model = WideResNet(depth=28, num_classes=100).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 180], gamma=0.1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self,centers,epoch):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (imgs, index, target) in enumerate(self.train_loader):
#            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            mse = torch.nn.MSELoss()
            outputs = []
            loss = 0
            mean = 0
            mean_post = 0
            for idx in range(self.num_imgs):
              data, target = imgs[idx].to(self.device), target.to(self.device)
              output = self.model(data)
              outputs.append(output)
#              print(torch.nn.functional.one_hot(target).size())
              p = F.softmax(output).detach()
              one_hot = torch.from_numpy(to_one_hot_vector(100,target.cpu()).astype(np.float32)).cuda()
              one_hot = torch.sum(p * one_hot,dim=-1).unsqueeze(-1)
         #     print(one_hot.size(), output.size())
              mean = mean + one_hot*output
              mean_post += one_hot
            mean /= mean_post
            for idx in range(self.num_imgs):
              if epoch == 0:
                loss += self.criterion(outputs[idx], target) + mse(outputs[idx], mean)
              else:
                loss += self.criterion(outputs[idx], target) + mse(outputs[idx], mean) #.cuda() / 2)

            loss /= self.num_imgs

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)
     #       param = 0.5 #self.optimizer.param_groups[0]['lr']
    #        for idx in range(data.size()[0]):
   #           if centers[index[idx]] is False:
  #              centers[index[idx]] = mean[idx].detach()
 #             else:
#                centers[index[idx]] = (1 - param)  * centers[index[idx]] + param * mean[idx].detach()
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total, centers

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, _, target) in enumerate(self.test_loader):
                data, target = data[0].to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        centers = torch.zeros(50000, 100).cuda()
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train(centers,epoch)
            centers = train_result[2] 
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()

