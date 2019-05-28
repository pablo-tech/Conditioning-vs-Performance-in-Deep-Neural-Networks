import argparse

import resnet
import fcn

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import shutil
from os import listdir
import os
import matplotlib.pyplot as plt
# import torchviz

parser = argparse.ArgumentParser(description="Jakub's hacked up resnet trainer")

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')

parser.add_argument('-e', '--analyze')

parser.add_argument("--make_dot", action='store_true', help="Create a dot file for vis", default=False)

parser.add_argument('--drop_p', type=float, help='the drop probability in dropout')
parser.add_argument('--ddrop_p', type=float, help='the drop probability in dropout')

parser.add_argument('--dest', default="")

def main():

    args = parser.parse_args()
    best_acc1 = 0

    if args.analyze is not None:
        analyze(args)
        exit()

    # We do not in fact care to pre-train
    if args.arch == "resnet18":
        model = resnet.resnet18(num_classes=200)

    if args.arch == "fudgeresnet18":
        model = resnet.fudgeresnet18(num_classes=200)

    if args.arch == "fcn":
        model = fcn.FullyConnected(batch_size=args.batch_size)

    if args.arch == "fcn_drop":
        if args.drop_p is not None:
            model = fcn.FullyConnectedSingleDrop(batch_size=args.batch_size, drop_p=args.drop_p)
        else:
            print("You forgot the drop probability")
            exit()

    if args.arch == "fcn_ddrop":
        if args.ddrop_p is not None:
            model = fcn.FullyConnectedDoubleDrop(batch_size=args.batch_size, drop_p=args.ddrop_p)
        else:
            print("You forgot the ddrop probability")
            exit()

    if args.arch == "fudge":
        model = fcn.FudgellyConnected(batch_size=args.batch_size)

    if args.make_dot:
        if args.arch is not None:
            x = torch.autograd.Variable(torch.randn(1, 256*224*224*3))
            print(model)
            y = model(x)
            # a = torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))

            torch.onnx.export(model, x, args.arch + ".onnx", verbose=True, input_names=["input_names"], output_names=["output_names"])
            # print(a)
            exit()
        else:
            print("You forgot to select an architecture")
            exit()


    if args.gpu is None:
        criterion = nn.CrossEntropyLoss()
    else:
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    traindir = "tiny-imagenet-200/train"
    valdir = "tiny-imagenet-200/val"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    # if we distribute this then this shouldn't be None.
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, drop_last=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("Getting ready to train!!!!")
    
    for epoch in range(args.start_epoch, args.epochs):
        filename = "checkpoint_" + str(epoch).zfill(3) + "_" + args.arch + ".pth.tar"
        print("Will save file to %s", filename)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_acc1': best_acc1,
        #     'curr_acc1': acc1,
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best, filename=filename)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'curr_acc1': acc1,
            'layer2_weight_grad': model.layer2.weight.grad,
            'layer3_weight_grad': model.layer3.weight.grad,
            # 'fc_weight_grad' : model.fc.weight.grad,
            'stats': analyze_iteration(model.state_dict())
        }, False, filename=args.dest+"/"+"conditioning_"+filename)
        print(acc1)


def analyze_layer(layer):
    data_dict = {}
    _, s, _ = torch.svd(layer)
    data_dict['singular_values'] = s
    data_dict['conditioning'] = max(s)/min(s)
    return data_dict

def analyze_iteration(model):
    data_dict = {}
    data_dict['layer2.weight'] = analyze_layer(model['layer2.weight'])
    data_dict['layer3.weight'] = analyze_layer(model['layer3.weight'])
    print(data_dict['layer2.weight']['conditioning'])
    print(data_dict['layer3.weight']['conditioning'])
    print(len(data_dict['layer2.weight']['singular_values']))
    print(data_dict['layer2.weight']['singular_values'].mean())
    print(data_dict['layer2.weight']['singular_values'].std())

    # data_dict['fc.weight'] = analyze_layer(model['fc.weight'])
    # print(data_dict['fc.weight']['conditioning'])
    # print(len(data_dict['fc.weight']['singular_values']))
    # print(data_dict['fc.weight']['singular_values'].mean())
    # print(data_dict['fc.weight']['singular_values'].std())
    return data_dict


def train(train_loader, model, criterion, optimizer, epoch, args):
    print("Entered the training function")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def box_plot_singular_values(data):
    fig, ax = plt.subplots()
    ax.set_title("Box Plots of Singular Values")
    box = []
    for d in data:
        box.append(d['stats']['layer3.weight']['singular_values'].numpy())
    ax.boxplot(box)
    plt.show(block=False)

    fig = plt.figure()
    plt.title('Eigen Value Distributions vs training epochs')
    plt.subplot(4, 1, 1)
    plt.hist(box[0:9])
    plt.xlabel('Eigen Values Epochs 0:9')

    plt.subplot(4, 1, 2)
    plt.hist(box[10:19])
    plt.xlabel('Eigen Values Epochs 9:19')

    plt.subplot(4, 1, 3)
    plt.hist(box[20:29])
    plt.xlabel('Eigen Values Epochs 20:29')

    plt.subplot(4, 1, 4)
    plt.hist(box[30:39])
    plt.show(block=False)
    plt.xlabel('Eigen Values Epochs 20:29')



def conditioning(data):
    cond_list = []
    mins_list = []
    maxs_list = []
    accuracy_list = []
    fig = plt.figure()
    for d in data:
        mins = min(d['stats']['layer3.weight']['singular_values'].numpy())
        maxs = max(d['stats']['layer3.weight']['singular_values'].numpy())
        cond = float(maxs)/float(mins)
        cond_list.append(cond)
        mins_list.append(mins)
        maxs_list.append(maxs)
        accuracy_list.append(d['curr_acc1'])

    plt.title('Singular Values vs Epochs')
    plt.subplot(3, 1, 1)
    plt.grid(True)
    plt.xlabel('Epoch #')
    plt.ylabel('Minimum singular value')
    plt.ylim(0, 0.005)
    plt.plot(mins_list, linestyle='None', marker=r'$\bowtie$')

    plt.subplot(3, 1, 2)
    plt.grid(True)
    plt.xlabel('Epoch #')
    plt.ylabel('Maximum singular value')
    plt.ylim(0, 1.5)
    plt.plot(maxs_list, linestyle='None', marker=r'$\bowtie$')

    plt.subplot(3, 1, 3)
    plt.grid(True)
    plt.xlabel('Epoch #')
    plt.ylabel('Conditioning')
    plt.yscale('symlog')
    # plt.ylim(10, 10e9)
    plt.plot(cond_list, linestyle='None', marker=r'$\bowtie$') 


    fig = plt.figure()

    diff_acc = [j-i for i, j in zip(accuracy_list[:-1], accuracy_list[1:])]

    plt.subplot(3, 1, 1)
    plt.grid(True)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.ylim(0, 15)
    plt.plot(accuracy_list, linestyle='None', marker=r'$\bowtie$')

    plt.subplot(3, 1, 2)
    plt.grid(True)
    plt.xlabel('Epoch #')
    plt.ylabel('Diff Accuracy')
    plt.ylim(-1.5, 1.5)
    plt.plot(diff_acc, linestyle='None', marker=r'$\bowtie$')

    plt.subplot(3, 1, 3)
    plt.grid(True)
    plt.xlabel('Epoch #')
    plt.ylabel('Conditioning')
    # plt.ylim(10, 10e5)
    plt.plot(cond_list, linestyle='None', marker=r'$\bowtie$') 

    plt.show(block=False)

    fig = plt.figure()
    plt.grid(True)
    plt.xscale('symlog')
    for i in range(0, len(cond_list)-1):
        plt.quiver(cond_list[i], accuracy_list[i],  cond_list[i+1] - cond_list[i], accuracy_list[i+1]- accuracy_list[i], angles='xy', scale_units='xy', scale=1)
    plt.show(block=False)

def analyze(args):
    data = load_folder(args)
    box_plot_singular_values(data)
    conditioning(data)
    input("Press Enter to continue...")

def load_folder(args):
    if os.path.isdir(args.analyze):
            print("Loading folder contents %s", args.analyze)
            models = listdir(args.analyze)
            models.sort()
            i = 0
            data = []
            for m in models:
                # no point in wasting GPU resources on this....
                data.append(torch.load(args.analyze + "/" + m, map_location=lambda storage, loc: storage))
                i = i + 1
            return data
            # analyze(data)
    else:
        print("No folder found at %s", args.analyze)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()