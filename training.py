import torch
from CNN import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



if __name__ == '__main__':
    model = CNN().to(DEVICE)