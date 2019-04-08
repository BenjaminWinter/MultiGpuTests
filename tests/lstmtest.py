"""
A series of speed tests on pytorch LSTMs.
- LSTM is fastest (no surprise)
- When you have to go timestep-by-timestep, LSTMCell is faster than LSTM
- Iterating using chunks is slightly faster than __iter__ or indexing depending on setup
"""

import argparse
import os
import sys
import time
import timeit

import torch
import torch.cuda
from torch import nn
from torch.autograd import Variable


def time_speed(args, model, cuda, number, backward=False):
    def run():
        if cuda:
            out = torch.cuda.FloatTensor()
        else:
            out = torch.FloatTensor()
        x = Variable(torch.randn(args.batch_len, args.batch_size, args.dim_in, out=out))
        h = model(x)
        if backward:
            h.sum().backward()

    run()
    elapsed = 1000. * timeit.timeit(run, number=number)
    return elapsed


def lstm_model(args, cuda):
    lstm = nn.LSTM(args.dim_in, args.dim_out, 3,dropout=0.2, bidirectional=True)
    if cuda:
        lstm.cuda()
        lstm = nn.DataParallel(lstm)

    def fun(x):
        h, state = lstm(x)
        return h

    return fun


def lstm_cell_model_iter(args, cuda):
    lstm = nn.LSTMCell(args.dim_in, args.dim_out)
    if cuda:
        lstm.cuda()

    def fun(x):
        n = x.size(1)
        h0 = Variable(x.data.new(n, args.dim_out).zero_())
        state = (h0, h0)
        hs = []
        for i in x:
            h, state = lstm(i, state)
            state = (h, state)
            hs.append(h)
        hs = torch.stack(hs, dim=0)
        return hs

    return fun


def lstm_cell_model_chunk(args, cuda):
    lstm = nn.LSTMCell(args.dim_in, args.dim_out)
    if cuda:
        lstm.cuda()

    def fun(x):
        n = x.size(1)
        h0 = Variable(x.data.new(n, args.dim_out).zero_())
        state = (h0, h0)
        hs = []
        for i in x.chunk(x.size(0), 0):
            h, state = lstm(i.squeeze(0), state)
            state = (h, state)
            hs.append(h)
        hs = torch.stack(hs, dim=0)
        return hs

    return fun


def lstm_cell_model_range(args, cuda):
    lstm = nn.LSTMCell(args.dim_in, args.dim_out)
    if cuda:
        lstm.cuda()

    def fun(x):
        n = x.size(1)
        h0 = Variable(x.data.new(n, args.dim_out).zero_())
        state = (h0, h0)
        hs = []
        for i in range(x.size(0)):
            h, state = lstm(x[i], state)
            state = (h, state)
            hs.append(h)
        hs = torch.stack(hs, dim=0)
        return hs

    return fun


def lstm_iterative_model(args, cuda):
    lstm = nn.LSTM(args.dim_in, args.dim_out)
    if cuda:
        lstm.cuda()

    def fun(x):
        state = None
        hs = []
        for i in x.chunk(x.size(0), 0):
            h, state = lstm(i, state)
            hs.append(h)
        hs = torch.cat(hs, dim=0)
        return hs

    return fun


def time_speeds(args, cuda, number):
    def timer(model_fn_name):
        model_fn = globals()[model_fn_name]
        stats = []
        if not args.no_forward:
            fwd = time_speed(args, model_fn(args, cuda), cuda, number, backward=False)
            stats.append("{:.6f}ms forward".format(fwd))
        if not args.no_forward:
            bwd = time_speed(args, model_fn(args, cuda), cuda, number, backward=True)
            stats.append("{:.6f}ms backward".format(bwd))
        print("{}: {}".format(model_fn_name, ", ".join(stats)))

    if not args.no_lstm:
        timer('lstm_model')
    if not args.no_lstm_cell_iter:
        timer('lstm_cell_model_iter')
    if not args.no_lstm_cell_range:
        timer('lstm_cell_model_range')
    if not args.no_lstm_cell_chunk:
        timer('lstm_cell_model_chunk')
    if not args.no_lstm_iterative:
        timer('lstm_iterative_model')


def run(args):
    print("OS: {}, pytorch version: {}".format(os.name, torch.__version__))
    if torch.cuda.is_available():
        from torch.backends import cudnn
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("Device: {}, CUDA: {}, CuDNN: {}".format(name, cudnn.cuda, cudnn.version()))
    print("Test setup: ({},{},{})->({},{},{})".format(
        args.batch_len, args.batch_size, args.dim_in,
        args.batch_len, args.batch_size, args.dim_out
    ))
    starttime = time.time()
    if (not args.no_gpu) and torch.cuda.is_available():
        print("GPU Results")
        time_speeds(args, cuda=True, number=args.gpu_number)

    if not args.no_cpu:
        print("CPU Results")
        time_speeds(args, cuda=False, number=args.cpu_number)
    endtime = time.time()
    elapsed = endtime - starttime
    res = "Testing took {} sec".format(elapsed)
    print(res)
    with open(args.logfile, 'a') as f:
        f.write(res)
        f.write('\n')


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Pytorch LSTM Speedtest')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--batch-len', type=int, default=200, metavar='N', help='batch len')
    parser.add_argument('--dim-in', type=int, default=40, metavar='N', help='in dim')
    parser.add_argument('--dim-out', type=int, default=256, metavar='N', help='out dim')

    parser.add_argument('--cpu-number', type=int, default=20, metavar='N', help='iterations on CPU')
    parser.add_argument('--gpu-number', type=int, default=100, metavar='N', help='iterations on GPU')

    parser.add_argument('--no-lstm', action='store_true', help='disable LSTM test')
    parser.add_argument('--no-lstm-cell-iter', action='store_true', help='disable LSTMCell with iterator test')
    parser.add_argument('--no-lstm-cell-range', action='store_true', help='disable LSTMCell with slicing test')
    parser.add_argument('--no-lstm-cell-chunk', action='store_true', help='disable LSTMCell with chunks test')
    parser.add_argument('--no-lstm-iterative', action='store_true', help='disable LSTM iterative test')

    parser.add_argument('--no-gpu', action='store_true', help='disable GPU tests')
    parser.add_argument('--no-cpu', action='store_true', help='disable CPU tests')

    parser.add_argument('--no-forward', action='store_true', help='disable forward tests')
    parser.add_argument('--no-backward', action='store_true', help='disable backward tests')
    parser.add_argument('--logfile', type=str,
                        default="log.txt", help='Log File Location')

    args = parser.parse_args(argv)
    assert not (args.no_forward and args.no_backward)
    assert not (args.no_gpu and args.no_cpu)
    return args


def main(argv):
    args = parse_args(argv)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
