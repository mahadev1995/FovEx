from torchvision.io import read_image
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from math import exp
import numpy as np
import random
import torch
import os


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(42)

class FovExWrapper(nn.Module):
    def __init__(self, downstream_model, criterion, target_function, image_size, foveation_sigma, blur_filter_size, blur_sigma, forgetting, heatmap_sigma, heatmap_forgetting, foveation_aggregation=1, device="cuda"):
        super(FovExWrapper, self).__init__()

        self.image_size = image_size
        self.blur_filter_size = blur_filter_size
        self.blur_sigma = blur_sigma
        self.foveation_sigma = foveation_sigma
        self.forgetting = forgetting
        self.foveation_aggregation = foveation_aggregation

        self.internal_representation = None
        self.ones = None
        self.device = device

        self.downstream_model = downstream_model
        self.criterion = criterion
        self.target_function = target_function

        self.heatmap_sigma = heatmap_sigma
        self.heatmap_forgetting = heatmap_forgetting

    def forward(self, x, foveation_positions):
        if self.internal_representation is None:
            raise Exception("First set internal representation with function: initialize_scanpath_generation()")
        foveation_area = get_foveation(self.foveation_aggregation, self.foveation_sigma, self.image_size, foveation_positions)
        current_foveation_area = self.internal_representation + foveation_area
        blurring_mask = torch.clip(self.ones - current_foveation_area, 0, 1)
        applied_blur = self.blur * blurring_mask

        output = self.downstream_model(x + applied_blur)

        return output

    def initialize_scanpath_generation(self, x, batch_size):
        self.internal_representation = torch.zeros((batch_size, 1, self.image_size, self.image_size), device='cuda')
        self.ones = torch.ones((batch_size, 1, self.image_size, self.image_size), device=self.device)
        self.blur = calculate_blur(x, self.blur_filter_size, self.blur_sigma)

    def run_optimization(self, x, labels, scanpath_length, opt_iterations, learning_rate, random_restarts=False):
        batch_size = x.size(0)
        targets = self.target_function(x, labels)
        self.initialize_scanpath_generation(x, batch_size)

        scanpath = []
        loss_history = []

        for _ in range(scanpath_length):
            foveation_pos = torch.zeros((batch_size, 2, 1, 1), device='cuda', requires_grad=True)
            best_foveation_pos = torch.zeros((batch_size, 2, 1, 1), device='cuda')  # * 2 - 1
            best_loss = torch.ones((batch_size), device='cuda', dtype=torch.float32) * float("inf")

            for _ in range(opt_iterations):
                output = self(x, foveation_pos)
                # calculate the loss
                loss = self.criterion(output, targets)
                total_loss = loss.mean()
                # backward pass: compute gradient of the loss with respect to model parameters
                grad = torch.autograd.grad(total_loss, foveation_pos)[0]
                # perform a single optimization step (parameter update)
                foveation_pos.data -= torch.sign(grad) * learning_rate
                # save best
                idxs = loss < best_loss
                best_loss[idxs] = loss[idxs]
                best_foveation_pos[idxs] = foveation_pos[idxs]
                if torch.sum(~idxs) > 0:
                    #randomize positions that are worse than in previous optimization step
                    if random_restarts:
                        foveation_pos.data[~idxs] = torch.rand_like(best_foveation_pos.data[~idxs]) * 2 - 1
                    else:
                        foveation_pos.data[~idxs] += torch.rand_like(best_foveation_pos.data[~idxs]) * learning_rate - learning_rate / 2
            # Update internal representation
            current_foveation_mask = get_foveation(self.foveation_aggregation, self.foveation_sigma, self.image_size, best_foveation_pos)
            self.internal_representation = (self.internal_representation * self.forgetting + current_foveation_mask).detach()
            blur_mask = self.ones - self.internal_representation
            # Save positions in array
            scanpath.append(best_foveation_pos.detach())
            # Loss history
            loss_history.append(loss.detach())
        return torch.stack(scanpath, 1).squeeze(), torch.stack(loss_history, 1).squeeze(), self.internal_representation

    def generate_explanation(self, x, labels, scanpath_length, opt_iterations, learning_rate, random_restarts=False, normalize_heatmap=True, seed=42):   
        set_seed(seed)   
        current_scanpaths, current_loss_history, internal_rep = self.run_optimization(x, 
                                                                                        labels, 
                                                                                        scanpath_length, 
                                                                                        opt_iterations, 
                                                                                        learning_rate, 
                                                                                        random_restarts)

        heatmap = get_heat_maps(self.heatmap_sigma, 
                                self.image_size, 
                                current_scanpaths[None],
                                self.heatmap_forgetting, 
                                self.device, 
                                normalize_heatmap)
        
        return heatmap, current_scanpaths, current_loss_history, internal_rep

def calc_gaussian(a, std_dev, image_size, positions):
    B = positions.shape[0]
    xa, ya = create_grid(B, image_size)
    xa = xa - positions[:, 0]
    ya = ya - positions[:, 1]
    distance = (xa**2 + ya**2)
    g = a * torch.exp(-distance / std_dev)
    return g.view(B, 1, image_size, image_size)

def create_grid(batch_size, size):
    t = torch.linspace(-1, 1, size).cuda()
    xa, ya = torch.meshgrid([t, t])
    xa = xa.view(1, size, size).repeat(batch_size, 1, 1)
    ya = ya.view(1, size, size).repeat(batch_size, 1, 1)
    return xa, ya

def calculate_blur(images, blur_filter_size, sigma=5):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel, sigma):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    window = create_window(blur_filter_size, 3, sigma).cuda()
    pad = nn.ReflectionPad2d(padding=blur_filter_size // 2)
    imgs_pad = pad(images)
    blured_images = F.conv2d(imgs_pad, window, groups=3)
    blur = blured_images - images
    return blur

def get_foveation(foveation_aggregation, foveation_sigma, image_size, positions):
    mask = calc_gaussian(foveation_aggregation, foveation_sigma, image_size, positions)
    return mask

def normalize(x):
    x = (x - x.min())/(x.max() - x.min())
    return x

def get_heat_maps(foveation_sigma, image_size, scanpaths, forgetting_factor, device, normalization):

    """
    Calculates the heat maps based on the given scanpaths using foveation.

    Args:
        foveation_sigma (float): The standard deviation of the foveation Gaussian kernel.
        image_size (int): The size of the square image in pixels.
        scanpaths (torch.Tensor): The tensor containing scanpaths.
        forgetting_factor (list): The factor determining the influence of previous foveation mask on the current foveation masks.
        device (str): The device on which the computation will be performed.
        normalize (bool): The flag which says whether to normalize the final heatmap or not

    Returns:
        torch.Tensor: The heat maps of shape (batch_size, 1, image_size, image_size).
    """
    batch_size = scanpaths.shape[0]
    heat_map = torch.zeros((batch_size, 1, image_size, image_size)).to(device)
   
    for i in range(scanpaths.shape[1]):
        current_foveation_mask = get_foveation(1, foveation_sigma,
                                               image_size,
                                               scanpaths[:, i, :, None, None],
                                               )

        heat_map = forgetting_factor[i]*heat_map + current_foveation_mask

    if normalization:
        heat_map = normalize(heat_map)

    return heat_map

def create_heatmap_reduce_max(scanpaths, image_size, sigma):
    batch_size = scanpaths.shape[0]
    heat_map = torch.zeros((batch_size, scanpaths.shape[1], image_size, image_size)).to(scanpaths.device)

    for i in range(scanpaths.shape[1]):
        heat_map[:, i] = get_foveation(1, 
                                       sigma,
                                       image_size,
                                       scanpaths[:, i, :, None, None],
                                       )

    heatmap = np.max(heat_map.cpu().numpy(), axis=1)
    
    return heatmap


def create_directory(path):
    import os
    if os.path.exists(path):
        print('Directory already Exists: ', path)
    else:
        os.makedirs(path)
        print("Directory  '%s'  created" %path)


def read_img(img_path):
    image = read_image(img_path)

    if image.shape[0] ==1:
        image = image.repeat(3, 1, 1)
        
    if image.shape[0] == 4:
        image = image[:3, ...]

    return image