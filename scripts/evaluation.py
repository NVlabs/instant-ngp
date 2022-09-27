import argparse

import numpy as np
from tqdm import tqdm

from scripts.common import compute_error
from swn.data import KITTI
from swn.metrics import KITTIMetrics


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluation of performance for Depth and Image Renderings")
	parser.add_argument("--path_raw_data", default="images", help="input path to the images depth and renderings")
	parser.add_argument("--path_depth_gt", default="images", help="gt path to the images depth and rgb")
	parser.add_argument("--path_output", default="test", help="Data to test")
	args = parser.parse_args()
	return args


#
# class SSIM(nn.Module):
#     """Layer to compute the SSIM loss between a pair of images
#     """
#     def __init__(self):
#         super(SSIM, self).__init__()
#         self.mu_x_pool   = nn.AvgPool2d(3, 1)
#         self.mu_y_pool   = nn.AvgPool2d(3, 1)
#         self.sig_x_pool  = nn.AvgPool2d(3, 1)
#         self.sig_y_pool  = nn.AvgPool2d(3, 1)
#         self.sig_xy_pool = nn.AvgPool2d(3, 1)
#
#         self.refl = nn.ReflectionPad2d(1)
#
#         self.C1 = 0.01 ** 2
#         self.C2 = 0.03 ** 2
#
#     def forward(self, x, y):
#         x = self.refl(x)
#         y = self.refl(y)
#
#         mu_x = self.mu_x_pool(x)
#         mu_y = self.mu_y_pool(y)
#
#         sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
#         sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
#         sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
#
#         SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
#         SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
#
#         return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

#
# import numpy
# from scipy import signal
# from scipy import ndimage
#
# import gauss
#
#
# def ssim(img1, img2, cs_map=False):
# 	"""Return the Structural Similarity Map corresponding to input images img1
# 	and img2 (images are assumed to be uint8)
#
# 	This function attempts to mimic precisely the functionality of ssim.m a
# 	MATLAB provided by the author's of SSIM
# 	https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
# 	"""
# 	img1 = img1.astype(numpy.float64)
# 	img2 = img2.astype(numpy.float64)
# 	size = 11
# 	sigma = 1.5
# 	window = gauss.fspecial_gauss(size, sigma)
# 	K1 = 0.01
# 	K2 = 0.03
# 	L = 255  # bitdepth of image
# 	C1 = (K1 * L) ** 2
# 	C2 = (K2 * L) ** 2
# 	mu1 = signal.fftconvolve(window, img1, mode='valid')
# 	mu2 = signal.fftconvolve(window, img2, mode='valid')
# 	mu1_sq = mu1 * mu1
# 	mu2_sq = mu2 * mu2
# 	mu1_mu2 = mu1 * mu2
# 	sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
# 	sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
# 	sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
# 	if cs_map:
# 		return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
# 															 (sigma1_sq + sigma2_sq + C2)),
# 				(2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
# 	else:
# 		return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
# 															(sigma1_sq + sigma2_sq + C2))


if __name__ == "__main__":
	args = parse_args()
	metrics = {"abs_rel": [],
					"sq_rel": [],
					"rmse": [],
					"rmse_log": [],
					"a1": [],
					"a2": [],
					"a3": [],
			   		"ssim":[]
					}
	kittimetrics = KITTIMetrics()
	for batch_idx, (x, y, x_nerf, y_nerf) in tqdm(enumerate(KITTI(args.path_depth_gt, args.path_raw_data, split_type="eigen_with_gt", split="test", path_nerf_renders=args.path_output))):
		errors = kittimetrics(y.unsqueeze(0), y_nerf.unsqueeze(0))
		ssim = float(compute_error("SSIM", x, x_nerf))

		metrics['ssim'].append(ssim)
		metrics['abs_rel'].append(errors['abs_rel'])
		metrics['sq_rel'].append(errors['sq_rel'])
		metrics['rmse'].append(errors['rmse'])
		metrics['rmse_log'].append(errors['rmse_log'])
		metrics['a1'].append(errors['a1'])
		metrics['a2'].append(errors['a2'])
		metrics['a3'].append(errors['a3'])

	print("ssim:{}".format(np.array(metrics['ssim']).mean()))
	print("Rmse:{}".format(np.array(metrics['rmse']).mean()))
	print("Abs_rel:{}".format(np.array(metrics['abs_rel']).mean()))
	print("Sq_rel:{}".format(np.array(metrics['sq_rel']).mean()))
	print("Rmse_log:{}".format(np.array(metrics['rmse_log']).mean()))
	print("A1:{}".format(np.array(metrics['a1']).mean()))
	print("A2:{}".format(np.array(metrics['a2']).mean()))
	print("A3:{}".format(np.array(metrics['a3']).mean()))
