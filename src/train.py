import numpy as np
import pandas as pd
import torch
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable

from src.preprocess import PreProcess
from src.model import VDACaps,CapsuleLoss


writer = SummaryWriter('VDACaps_log')

NUM_CLASSES = 7
data_path = '.\CK+'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=data_path)
parser.add_argument('--num_classes', default=7, type=int, metavar='N', help='expression category')
parser.add_argument('--n_epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--num_routing_iterations', default=3, type=int, metavar='N', help='The actual number of iterations is 2')
parser.add_argument('-b', '--batchsz', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', dest='lr')
parser.add_argument('--num_workers1', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--num_workers2', default=2, type=int, metavar='N', help='number of data loading workers')
args = parser.parse_args()


vdacaps = VDACaps()
capsule_loss = CapsuleLoss()
optimizer = Adam(vdacaps.parameters(), lr=args.lr)

class average_meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

train_db = PreProcess(args.data, 224, mode='train')
val_db = PreProcess(args.data, 224, mode='val')
train_loader = DataLoader(train_db, batch_size=args.batchsz, shuffle=True,num_workers=args.num_workers1)
val_loader = DataLoader(val_db, batch_size=args.batchsz, num_workers=args.num_workers2)

def train(epoch):
    losses = average_meter()
    accuracy = average_meter()
    vdacaps.train()


    for batch_id, (data, label) in enumerate(train_loader):

        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=label)
        data, target = Variable(data).to(device), Variable(target).to(device)
        masked, output, reconstructions = vdacaps(data)

        data = data[:, 0:3, :, :]
        data = data.type(torch.FloatTensor)
        data = data
        loss = capsule_loss(data, target, output, reconstructions)
        losses.update(loss.item(), data.size(0))

        pred = output.data.max(1)[1]
        label = label.type(torch.LongTensor)
        label = Variable(label)
        prec = pred.eq(label.data).cpu().sum()

        accuracy.update(float(prec) / data.size(0), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch:%d   batchId:%d"%(epoch, batch_id))
        print("train loss:{},trian accuracy:{}".format(losses.val, accuracy.val))
        if (batch_id+1) % 100 == 0:
            print('Epoch [%d/%d], Iter[%d/%d], train loss. %.4f train accuracy: %.4f' %
                  (epoch, args.n_epochs, batch_id + 1, len(train_loader),
                  losses.val, accuracy.val))

        writer.add_scalar('Train/Loss', losses.val, epoch)
        writer.add_scalar('Train/Acc', accuracy.val, epoch)


def test(epoch):
    losses = average_meter()
    accuracy = average_meter()

    vdacaps.eval()

    mat = np.zeros((1, 7))
    for data, label in val_loader:
        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=label)

        data, target = Variable(data).to(device), Variable(target).to(device)

        masked, output, reconstructions = vdacaps(data)

        data = data[:, 0:3, :, :]
        data = data.type(torch.FloatTensor)
        data = data
        loss = capsule_loss(data, target, output, reconstructions)
        losses.update(loss.item(), data.size(0))

        pred = output.data.max(1)[1]
        label = label.type(torch.LongTensor)
        label = Variable(label)
        prec = pred.eq(label.data).cpu().sum()

        accuracy.update(float(prec) / data.size(0), data.size(0))

    df = pd.DataFrame(mat[1:])

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        losses.avg, int(accuracy.sum), len(val_loader), 100. * accuracy.avg))

    writer.add_scalar('Test/Loss', losses.avg, epoch)
    writer.add_scalar('Test/Acc', accuracy.avg, epoch)

    return accuracy.avg

#========================main========================#
def main():

    best_model = vdacaps.to(device)
    best_accuray = 0.0

    for epoch in range(1, args.n_epochs):

        train(epoch)
        val_accuracy = test(epoch)

        if best_accuray < val_accuracy:
            best_model = vdacaps
            best_accuray = val_accuracy

    writer.close()

    print("The best model has an accuracy of " + str(best_accuray))
    torch.save(best_model.state_dict(), './CK+.mdl')


if __name__ == '__main__':
    main()