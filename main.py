import argparse
from run import train


def parse_option():
    parser = argparse.ArgumentParser()

    # default settings
    parser.add_argument('--save', type=str, default='MuP')
    parser.add_argument('--dataset', type=str, default='allenai/mup')

    # training
    parser.add_argument('--num_epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--init_lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--train_checkpoint', type=int, default=100)
    parser.add_argument('--eval_checkpoint', type=int, default=1000)
    parser.add_argument('--dropout_p', type=float, default=0.5)

    # pretrain model settings
    parser.add_argument('--model_config', type=str, default="t5-base")
    parser.add_argument('--max_length', type=int, default=512)

    # distributed training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    config = parser.parse_args()
    return config


def main():
    config = parse_option()
    train(config)


if __name__ == '__main__':

    main()