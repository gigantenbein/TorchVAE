import argparse
import time
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import VAE, ConvolutionalVAE


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recon_loss', type=str, default='BCE', metavar='loss_type',
                    help='should the reconstruction loss be BCE or MSE')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

cifar_dataset = datasets.CIFAR10('./data', train=True, download=False,
                                 transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    cifar_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    cifar_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)


model = ConvolutionalVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, recon_type='BCE'):

    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3072), reduction='sum')
    MSE = F.mse_loss(recon_x, x.view(-1, 3072), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if recon_type == 'BCE':
        return BCE + KLD
    else:
        return MSE + KLD


def train(epoch):
    model.train()
    train_loss = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1)

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, recon_type='MSE')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # mse_loss = nn.MSELoss(reduction='mean')
        # mse_loss_ = mse_loss(recon_batch, data.view(-1, 3072))
        # scheduler.step(mse_loss_)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, recon_type='MSE').item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 3, 32, 32)[:n]])
                save_image(comparison.cpu(),
                         'results_c/reconstruction_' + timestring + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    timestamp = time.localtime()
    timestring = "_{}_{}_{}_{}_".format(timestamp.tm_year, timestamp.tm_mon, timestamp.tm_mday, timestamp.tm_hour)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 128).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 32, 32),
                       'results_c/sample_' + timestring + str(epoch) + '.png')