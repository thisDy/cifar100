import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ViTMLPModel
from models.Fed import FedAvg
from models.test import test_img

if __name__ == '__main__':
    args = args_parser()
    print("All arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    args.device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    print(args.device)

    # Dataset loading
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=transform)
        # Assigning dict_users based on IID setting
        dict_users = mnist_iid(dataset_train, args.num_users) if args.iid else mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform)
        # Ensuring CIFAR-10 is handled for IID setting only
        if not args.iid:
            print('Warning: CIFAR-10 non-IID setting not implemented. Defaulting to IID.')
        dict_users = cifar_iid(dataset_train, args.num_users)  # Always use IID for CIFAR-10 for now
    else:
        exit('Error: unrecognized dataset')

    # Ensure dict_users is not None
    if dict_users is None:
        exit('Error: Failed to initialize user dictionary. This should not happen.')


    # Model selection
    img_size = dataset_train[0][0].shape
    if args.model == 'vit_mlp':
        net_glob = ViTMLPModel(args).to(args.device)
    elif args.model == 'mlp':
        len_in = np.prod(img_size)
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    #print(net_glob)
    net_glob.train()

    # Training
    loss_train = []
    acc_train = []
    loss_test_list = []
    acc_test_list = []

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1} started')
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # Evaluation for each epoch
        net_glob.eval()
        #net_glob.to('cpu')  # Move model to CPU
        acc_train_epoch, loss_train_epoch = test_img(net_glob, dataset_train, args)
        acc_test_epoch, loss_test_epoch = test_img(net_glob, dataset_test, args)
        acc_train.append(acc_train_epoch)
        loss_test_list.append(loss_test_epoch)
        acc_test_list.append(acc_test_epoch)
        #net_glob.to(args.device)

        print(f'Epoch {epoch+1}, Training Loss: {loss_avg:.3f}, Training Accuracy: {acc_train_epoch:.2f}, Testing Accuracy: {acc_test_epoch:.2f}')

    # Plot Training and Testing Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_train)), loss_train, label='Training Loss')
    plt.plot(range(len(loss_test_list)), loss_test_list, label='Testing Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_testing_loss.png')

    # Plot Training and Testing Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(acc_train)), acc_train, label='Training Accuracy')
    plt.plot(range(len(acc_test_list)), acc_test_list, label='Testing Accuracy')
    plt.title('Training and Testing Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_testing_accuracy.png')
