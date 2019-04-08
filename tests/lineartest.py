"""
Optional: Data Parallelism
==========================
**Authors**: `Sung Kim <https://github.com/hunkim>`_ and `Jenny Kang <https://github.com/jennykang>`_

In this tutorial, we will learn how to use multiple GPUs using ``DataParallel``.

It's very easy to use GPUs with PyTorch. You can put the model on a GPU:

.. code:: python

    device = torch.device("cuda:0")
    model.to(device)

Then, you can copy all your tensors to the GPU:

.. code:: python

    mytensor = my_tensor.to(device)

Please note that just calling ``my_tensor.to(device)`` returns a new copy of
``my_tensor`` on GPU instead of rewriting ``my_tensor``. You need to assign it to
a new tensor and use that tensor on the GPU.

It's natural to execute your forward, backward propagations on multiple GPUs.
However, Pytorch will only use one GPU by default. You can easily run your
operations on multiple GPUs by making your model run parallelly using
``DataParallel``:

.. code:: python

    model = nn.DataParallel(model)

That's the core behind this tutorial. We will explore it in more detail below.
"""


######################################################################
# Imports and parameters
# ----------------------
#
# Import PyTorch modules and define parameters.
#

import torch, time, argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='linear multigpu test')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--hs', type=int, default=512, help='input size')
parser.add_argument('--logfile', type=str, default="log.txt", help='Log File Location')

args = parser.parse_args()


# Parameters and DataLoaders
input_size = args.hs
output_size = args.hs

batch_size = args.bs
data_size = 10000


######################################################################
# Device
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Dummy DataSet
# -------------
#
# Make a dummy (random) dataset. You just need to implement the
# getitem
#

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


######################################################################
# Simple Model
# ------------
#
# For the demo, our model just gets an input, performs a linear operation, and
# gives an output. However, you can use ``DataParallel`` on any model (CNN, RNN,
# Capsule Net etc.)
#
# We've placed a print statement inside the model to monitor the size of input
# and output tensors.
# Please pay attention to what is printed at batch rank 0.
#

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(3)])

    def forward(self, input):
        output = input
        for l in self.fc:
            output = l(output)

        return output


######################################################################
# Create Model and DataParallel
# -----------------------------
#
# This is the core part of the tutorial. First, we need to make a model instance
# and check if we have multiple GPUs. If we have multiple GPUs, we can wrap
# our model using ``nn.DataParallel``. Then we can put our model on GPUs by
# ``model.to(device)``
#

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)


######################################################################
# Run the Model
# -------------
#
# Now we can see the sizes of input and output tensors.
#
start = time.time()
i = 0
for i in range(100):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        i += 1
res = "Testing took {} sec".format(i, time.time() - start) 
print(res)
with open(args.logfile, 'a') as f:
    f.write(res)
    f.write('\n')