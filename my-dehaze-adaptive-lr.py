import datetime
import os
import sys
from collections import namedtuple
import time
import numpy as np

import torch.nn as nn
from cv2.ximgproc import guidedFilter
from net import *
from net.losses import StdLoss
from net.noise import get_noise
from skimage.measure import compare_psnr
from utils.image_io import *
from utils.imresize import imresize, np_imresize

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
DehazeResult = namedtuple("DehazeResult", ['learned', 't', 'a', 'psnr', 'step'])


def dehaze(image_name: str, image: np.array, num_iter: int = 4000,
           plot_during_training: bool = True, show_every: int = 500,
           scorefile: str = None):
    dehazer = Dehazer(image_name, image, num_iter, plot_during_training, show_every, scorefile)
    dehazer.optimize()
    dehazer.finalize()
    save_image(image_name + "_original", np.clip(image, 0, 1))


class EarlyStopping(object):
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def step(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            return False


class Dehazer(object):
    def __init__(self, image_name, image, num_iter=8000, plot_during_training=True, show_every=500, scorefile=None):
        self.image_name = image_name
        self.image = image
        self.num_iter = num_iter
        self.plot_during_training = plot_during_training
        self.show_every = show_every
        self.scorefile = scorefile
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None

        self.learning_rate = 0.001
        self.input_depth = 8
        self.parameters = None
        self.current_result = None
        self.best_result = None
        self.break_iter = None
        self.time_elaps = None
        self.step_early = None
        self.optimizer = None

        self.mse_loss = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
        self.blur_loss = StdLoss().type(torch.cuda.FloatTensor)
        self.total_loss = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        if scorefile:
            self._init_scorefile()

    def _init_images(self):
        self.original_image = self.image.copy()
        image = self.image
        factor = 1
        while image.shape[1] >= 800 or image.shape[2] >= 800:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] / factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1
        self.images = create_augmentations(image)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor

        image_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.image_net = image_net.type(data_type)

        mask_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.mask_net = mask_net.type(data_type)

        ambient_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.ambient_net = ambient_net.type(data_type)

    def _init_ambient(self):
        atmosphere = self.get_atmosphere(self.image)
        self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                        requires_grad=False)

    def _init_parameters(self):
        self.parameters = [p for p in self.image_net.parameters()] + \
                          [p for p in self.mask_net.parameters()] + \
                          [p for p in self.ambient_net.parameters()]

    def _init_inputs(self):
        original_noises = create_augmentations(torch_to_np(get_noise(self.input_depth, 'noise',
                                                                     (self.images[0].shape[1], self.images[0].shape[2]),
                                                                     var=1 / 10.)
                                                           .type(torch.cuda.FloatTensor).detach()))
        self.image_net_inputs = [np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()
                                 for original_noise in original_noises]

        original_noises = create_augmentations(torch_to_np(get_noise(self.input_depth, 'noise',
                                                                     (self.images[0].shape[1], self.images[0].shape[2]),
                                                                     var=1 / 10.).type(
            torch.cuda.FloatTensor).detach()))

        self.mask_net_inputs = [np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()
                                for original_noise in original_noises]

        self.ambient_net_input = get_noise(self.input_depth, 'meshgrid',
                                           (self.images[0].shape[1], self.images[0].shape[2])
                                           ).type(torch.cuda.FloatTensor).detach()

    def _init_scorefile(self):
        if not os.path.isfile(self.scorefile):
            with open(self.scorefile, 'a') as file:
                file.write("filename,psnr,time,best_result_step,early_stop_at\n")

    def get_dark_channel(self, image, w=15):
        """
        Get the dark channel prior in the (RGB) image data.
        Parameters
        -----------
        image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
            M is the height, N is the width, 3 represents R/G/B channels.
        w:  window size
        Return
        -----------
        An M * N array for the dark channel prior ([0, L-1]).
        """
        M, N, _ = image.shape
        padded = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
        darkch = np.zeros((M, N))
        for i, j in np.ndindex(darkch.shape):
            darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
        return darkch

    def get_atmosphere(self, image, p=0.0001, w=15):
        """Get the atmosphere light in the (RGB) image data.
        Parameters
        -----------
        image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
        w:      window for dark channel
        p:      percentage of pixels for estimating the atmosphere light
        Return
        -----------
        A 3-element array containing atmosphere light ([0, L-1]) for each channel
        """
        image = image.transpose(1, 2, 0)
        # reference CVPR09, 4.4
        darkch = self.get_dark_channel(image, w)
        M, N = darkch.shape
        flatI = image.reshape(M * N, 3)
        flatdark = darkch.ravel()
        searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
        # return the highest intensity for each channel
        return np.max(flatI.take(searchidx, axis=0), axis=0)

    def optimize(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

        es = EarlyStopping(patience=2)
        tic = time.perf_counter()
        for step in range(self.num_iter):
            self.optimizer.zero_grad()
            self._optimization_step(step)

            self.cur_step = step
            if step % 8 == 0:
                self._obtain_current_result()
            if self.plot_during_training:
                self._plot_current(step)
            if es.step(self.total_loss.item()):
                self.step_early = step
                break

            self.optimizer.step()

        toc = time.perf_counter()
        self.time_elaps = toc - tic

    def _optimization_step(self, step):
        """

        :param step: the number of the iteration

        :return:
        """
        # if num of itereations reached
        if step == self.num_iter - 1:
            reg_std = 0
        else:
            reg_std = 1 / 30.
        aug = 0
        image_net_input = self.image_net_inputs[aug] + (self.image_net_inputs[aug].clone().normal_() * reg_std)
        self.image_out = self.image_net(image_net_input)

        ambient_net_input = self.ambient_net_input + (self.ambient_net_input.clone().normal_() * reg_std)
        self.ambient_out = self.ambient_net(ambient_net_input)

        self.mask_out = self.mask_net(self.mask_net_inputs[aug])

        self.blur_out = self.blur_loss(self.mask_out)
        self.total_loss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                        self.images_torch[aug]) + 0.005 * self.blur_out
        self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
        if step < 1000:
            self.total_loss += self.mse_loss(self.ambient_out, self.ambient_val * torch.ones_like(self.ambient_out))
        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self):
        image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
        mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
        ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
        psnr = compare_psnr(self.images[0], mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np)
        self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np, psnr=psnr,
                                           step=self.cur_step)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.counter = 0
            self.best_result = self.current_result

    def _plot_current(self, step):
        """

         :param step: the number of the iteration

         :return:
         """
        print('Iteration %05d    Loss %f Lr %f %f current_psnr: %f max_psnr %f' % (step, self.total_loss.item(),
                                                                                   self.optimizer.param_groups[0]['lr'],
                                                                                   self.blur_out.item(),
                                                                                   self.current_result.psnr,
                                                                                   self.best_result.psnr), '\r', '')
        if step % self.show_every == self.show_every - 1:
            plot_image_grid("t_and_amb",
                            [self.best_result.a * np.ones_like(self.best_result.learned), self.best_result.t])
            plot_image_grid("current_image", [self.images[0], np.clip(self.best_result.learned, 0, 1)])

    def finalize(self):
        self.final_image = np_imresize(self.best_result.learned, output_shape=self.original_image.shape[1:])
        self.final_t_map = np_imresize(self.best_result.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.best_result.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.t_matting(self.final_t_map)
        post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
        save_image(self.image_name + "_original", np.clip(self.original_image, 0, 1))
        save_image(self.image_name + "_learned", self.final_image)
        save_image(self.image_name + "_t", mask_out_np)
        save_image(self.image_name + "_final", post)
        save_image(self.image_name + "_a", np.clip(self.final_a, 0, 1))
        if self.scorefile:
            with open(self.scorefile, 'a') as file:
                file.write(self.image_name + "," + str(self.best_result.psnr) + "," + str(self.time_elaps) + ","
                           + str(self.best_result.step) + "," + str(self.step_early) + "\n")

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        return np.array([np.clip(refine_t, 0.1, 1)])


if __name__ == "__main__":
    scorepath = "output/scores/"
    if not os.path.exists(scorepath):
        os.makedirs(scorepath)
    scorefilename = "scores_" + str(datetime.datetime.now()) + ".csv"
    scorefile = scorepath + scorefilename

    if os.path.isfile(sys.argv[1]):
        image = prepare_image(sys.argv[1])
        dehaze(os.path.basename(sys.argv[1]), image)

    if os.path.isdir(sys.argv[1]):
        for file in os.listdir(os.fsencode(sys.argv[1])):
            filename = os.fsdecode(file)
            image = prepare_image(sys.argv[1] + filename)
            dehaze(os.path.basename(filename), image, num_iter=4000,
                   plot_during_training=True, scorefile=scorefile)

