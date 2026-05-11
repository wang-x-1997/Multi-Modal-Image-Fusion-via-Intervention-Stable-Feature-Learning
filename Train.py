import math
import os

import torch
from torch import nn


import torch.nn.functional as F
import numpy as np
from torch.utils import data as data

from typing import Tuple, Dict, Optional, List


class CausalInvarianceLoss(nn.Module):
    def __init__(self, alpha=0.1, eta: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.eta = eta
        self.eps = eps

    @staticmethod
    def _spatial_mean(g: torch.Tensor) -> torch.Tensor:
        return g.mean(dim=[2, 3], keepdim=True)

    @staticmethod
    def _entropy(g: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(g, 1e-6, 1 - 1e-6)
        ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return ent.mean(dim=[2, 3])

    def forward(self, y_base: torch.Tensor, y_do_list: List[torch.Tensor],
                g_list: List[torch.Tensor]) -> torch.Tensor:
        assert len(y_do_list) == len(g_list)
        total_loss = 0.0
        num = len(y_do_list)

        for y_do, g in zip(y_do_list, g_list):
            diff = torch.abs(y_do - y_base)
            consistency = (diff * g).mean()

            g_mean = self._spatial_mean(g)
            area_term = torch.abs(g_mean - self.eta).mean()
            entropy_term = self._entropy(g).mean()
            reg = area_term - entropy_term

            total_loss += consistency + reg

        return total_loss / num

class CausalNecessityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_full: torch.Tensor, y_vis_only: torch.Tensor, y_ir_only: torch.Tensor):
        loss_vis = F.l1_loss(y_full, y_vis_only)
        loss_ir = F.l1_loss(y_full, y_ir_only)
        return loss_vis + loss_ir

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Laplacian()
        # self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        # Loss_gradient = F.l1_loss(gradient_fused, gradient_A) +F.l1_loss(gradient_fused, gradient_B)
        return Loss_gradient

from loss_SSIM import *
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class Laplacian(nn.Module):
    def __init__(self, kernel_type='8-neighbor'):
        super(Laplacian, self).__init__()

        if kernel_type == '4-neighbor':
            # 4邻域拉普拉斯核
            kernel = [[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]]
        elif kernel_type == '8-neighbor':
            kernel = [[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]]
        elif kernel_type == 'alternative':
            kernel = [[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]]
        else:
            raise ValueError("kernel_type must be '4-neighbor', '8-neighbor', or 'alternative'")

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x):
        laplacian = F.conv2d(x, self.weight, padding=1)
        return torch.abs(laplacian)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        w1 = image_A/(image_B+image_A+1e-10)
        w2 = 1-w1
        # intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(w1 *image_fused, w1 *image_A) + F.l1_loss(w2 *image_fused, w2 *image_B)
        return Loss_intensity

class L_segexp(nn.Module):

    def __init__(self):
        super(L_segexp, self).__init__()

    def forward(self, x, y):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        # mean = self.pool(x)
        a1 = torch.zeros(b, h, w).cuda()
        d = 0
        for i in range(9):
            a2 = torch.where(y == i, x, a1).cuda()
            count = torch.sum(y == i).item()
            d2 = torch.sum(a2)
            if count:
                d2 = d2 / count

            # non_zero_count = torch.nonzero(a2).size(0)
            #
            # print('non_zero_count', non_zero_count)
            a2 = torch.where(a2 == 0, d2, a2).cuda()
            d3 = torch.mean(torch.pow(a2 - torch.FloatTensor([d2]).cuda(), 2))
            # print(d3.item())
            d = d + d3

        # d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d/9

import kornia
class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        # self.L_SSIM = L_SSIM()
        self.L_SSIM = kornia.losses.SSIMLoss(3, reduction='mean')
        # self.L_segexp = L_segexp()

        # print(1)

    def forward(self, image_A, image_B, image_fused):
        # loss_l1 = 1* self.L_Inten(image_A, image_B, image_fused)
        loss_l1 = F.l1_loss(image_fused,image_A) + F.l1_loss(image_fused,image_B)
        loss_gradient =self.L_Grad(image_A, image_B, image_fused)
        # loss_SSIM = (1-self.L_SSIM(image_A, image_B,image_fused) )
        # loss_SSIM =  self.L_SSIM(image_A, image_fused) + self.L_SSIM(image_B, image_fused)
        fusion_loss = loss_l1 + loss_gradient
        return fusion_loss

def img2tensor(imgs, bgr2rgb=True, float32=True):

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        # print('imgs.shape',imgs.shape)
        return _totensor(imgs, bgr2rgb, float32)

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

import cv2
import random
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf"), followlinks=True):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=followlinks)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


def random_crop(img, out_size):
    h, w = img.shape[:2]
    rnd_h = random.randint(0, h - out_size)
    rnd_w = random.randint(0, w - out_size)
    return img[rnd_h: rnd_h + out_size, rnd_w: rnd_w + out_size]

class FusionDataset2(data.Dataset):
    def __init__(self,train=False):
        super(FusionDataset2, self).__init__()
        self.train=train

        self.ir_folder = r"./IVIF_data/1/"
        self.vi_folder = r"./IVIF_data/1/"

        self.ir_paths = make_dataset(self.ir_folder)
        self.vi_paths = make_dataset(self.vi_folder)

        assert len(self.ir_paths) == len(self.vi_paths), "IR 和 VI 图像数量不一致"

    def __getitem__(self, index):
        ir_path = self.ir_paths[index]
        vi_path = self.vi_paths[index]

        try:
            img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            img_vi = cv2.imread(vi_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

            # 扩展维度为 (H, W, 1)
            if img_ir.ndim == 2:
                img_ir = np.expand_dims(img_ir, axis=2)
            if img_vi.ndim == 2:
                img_vi = np.expand_dims(img_vi, axis=2)

        except Exception as e:
            print(f"Error reading image at {ir_path} or {vi_path}: {e}")
            return self.__getitem__((index + 1) % len(self.ir_paths))

        if self.train:
            gt_size = 256

            img_ir, img_vi = random_crop(img_ir, gt_size),random_crop(img_vi, gt_size)

            img_ir, img_vi = augment([img_ir, img_vi], True, True)

        img_ir = img2tensor(img_ir, bgr2rgb=False, float32=True)
        img_vi = img2tensor(img_vi, bgr2rgb=False, float32=True)

        return img_ir,img_vi
    def __len__(self):
        return len(self.ir_paths)




class Encoder(nn.Module):

    def __init__(self, dim: int = 32):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(1, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.down1 = nn.MaxPool2d(2)

        self.module2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.down2 = nn.MaxPool2d(2)

        self.module3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, img):
        x1 = self.module1(img)        # [N,dim,H,W]
        x2 = self.down1(x1)
        x2 = self.module2(x2)         # [N,dim,H/2,W/2]
        x3 = self.down2(x2)
        x3 = self.module3(x3)         # [N,dim,H/4,W/4]
        return x1, x2, x3


def norm_1(x):
    n, c, h, w = x.shape
    x_flat = x.view(n, -1)
    max1 = x_flat.max(dim=1, keepdim=True)[0].view(n, 1, 1, 1)
    min1 = x_flat.min(dim=1, keepdim=True)[0].view(n, 1, 1, 1)
    return (x - min1) / (max1 - min1 + 1e-8)

class CFI(nn.Module):
    def __init__(self, dim: int, reduce: int = 8, q_chunk: int = 0):
        super().__init__()
        self.dim = dim
        self.reduce = max(1, int(reduce))
        self.q_chunk = int(q_chunk)

        self.q_v = nn.Conv2d(dim, dim, 1)
        self.k_v = nn.Conv2d(dim, dim, 1)
        self.v_v = nn.Conv2d(dim, dim, 1)

        self.q_i = nn.Conv2d(dim, dim, 1)
        self.k_i = nn.Conv2d(dim, dim, 1)
        self.v_i = nn.Conv2d(dim, dim, 1)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, padding=1),
            nn.BatchNorm2d(dim//2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim//2, 1, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def _pooled_xattn(self, q, k, v):
        n, c, h, w = q.shape
        rh = self.reduce
        rw = self.reduce

        k_red = F.adaptive_avg_pool2d(k, (rh, rw))  # [N,C,rh,rw]
        v_red = F.adaptive_avg_pool2d(v, (rh, rw))  # [N,C,rh,rw]

        hw = h * w
        hwr = rh * rw

        qf = q.view(n, c, hw)          # [N,C,HW]
        kf = k_red.view(n, c, hwr)     # [N,C,HWr]
        vf = v_red.view(n, c, hwr)     # [N,C,HWr]

        scale = 1.0 / math.sqrt(c)

        if self.q_chunk and self.q_chunk < hw:
            out = qf.new_zeros(n, c, hw)
            for s in range(0, hw, self.q_chunk):
                e = min(hw, s + self.q_chunk)
                q_chunk = qf[:, :, s:e]  # [N,C,chunk]
                att = torch.softmax(
                    torch.bmm(q_chunk.transpose(1, 2), kf) * scale,
                    dim=-1
                )                        # [N,chunk,HWr]
                out_chunk = torch.bmm(vf, att.transpose(1, 2))  # [N,C,chunk]
                out[:, :, s:e] = out_chunk
            out = out.view(n, c, h, w)
        else:
            att = torch.softmax(
                torch.bmm(qf.transpose(1, 2), kf) * scale,
                dim=-1
            )                            # [N,HW,HWr]
            out = torch.bmm(vf, att.transpose(1, 2))   # [N,C,HW]
            out = out.view(n, c, h, w)

        return out

    def forward(self, f_vi, f_ir):
        qv, kv, vv = self.q_v(f_vi), self.k_v(f_vi), self.v_v(f_vi)
        qi, ki, vi = self.q_i(f_ir), self.k_i(f_ir), self.v_i(f_ir)

        vi2ir = self._pooled_xattn(qv, ki, vi)
        ir2vi = self._pooled_xattn(qi, kv, vv)  

        cross = (vi2ir + ir2vi)  # [N,C,H,W]

        local = (f_vi + f_ir)

        g = (self.gate(cross))  # [N,1,H,W]
        fused = g * cross + (1.0 - g) * local

        fused = self.refine(fused)
        return fused, g

class Network(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.encoder_ir = Encoder(dim=dim)
        self.encoder_vis = Encoder(dim=dim)

        self.adjust_channels3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.adjust_channels4 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.adjust_channels5 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.module4 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),

        )

        self.module5 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.final_decoder = nn.Sequential(
            nn.Conv2d(dim, 1, 3, padding=1),
            nn.Sigmoid()

        )

        self.cfi3 = CFI(dim)
        self.cfi4 = CFI(dim)
        self.cfi5 = CFI(dim)

    def forward(self, vi, ir):
        v1, v2, v3 = self.encoder_vis(vi)  # [N,dim,H,...]
        i1, i2, i3 = self.encoder_ir(ir)

        f3, g3 = self.cfi3(v3, i3)
        x3 =  f3                              

        x4_up = F.interpolate(x3, size=v2.shape[-2:], mode='bilinear', align_corners=False)
        f4, g4 = self.cfi4(v2, i2)
        x4 = x4_up  + f4   # [N,3dim,H/2,W/2]
        x4 = self.adjust_channels4(x4)             # [N,dim,H/2,W/2]

        x5_up = F.interpolate(x4, size=v1.shape[-2:], mode='bilinear', align_corners=False)
        f5, g5 = self.cfi5(v1, i1)
        x5 = x5_up + f5   # [N,3dim,H,W]
        x5 = self.adjust_channels5(x5)             # [N,dim,H,W]

        out = (self.final_decoder(x5) )            # [N,1,H,W] in [0,1]

        intermediate_outputs = (x3, x4, x5)
        gates = {'g3': g3, 'g4': g4, 'g5': g5}

        return out, intermediate_outputs, gates


class InterventionMasker:
    def __init__(self, hole_size=(16, 16), num_holes=(1, 6), p_rect=0.7):
        self.hole_size = hole_size
        self.num_holes = num_holes
        self.p_rect = p_rect

    @staticmethod
    def _random_mask(n, h, w, num, hole_size, p_rect, device):
        mask = torch.ones(n, 1, h, w, device=device)

        yy_full, xx_full = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )

        for i in range(n):
            k = random.randint(num[0], num[1])
            for _ in range(k):
                if random.random() < p_rect:
                    # axis-aligned rectangle
                    hh = random.randint(hole_size[0], hole_size[1])
                    ww = random.randint(hole_size[0], hole_size[1])
                    y0 = random.randint(0, max(0, h - hh))
                    x0 = random.randint(0, max(0, w - ww))
                    mask[i, 0, y0:y0 + hh, x0:x0 + ww] = 0.
                else:
                    # disk
                    rr = random.randint(hole_size[0] // 2, hole_size[1] // 2)
                    cy = random.randint(rr, h - rr) if h - rr > rr else rr
                    cx = random.randint(rr, w - rr) if w - rr > rr else rr
                    circle = ((yy_full - cy) ** 2 + (xx_full - cx) ** 2) <= (rr * rr)
                    mask[i, 0][circle] = 0.

        return mask

    def complementary(self, vi, ir):
        n, _, h, w = vi.shape
        device = vi.device

        m_vi = self._random_mask(n, h, w, self.num_holes, self.hole_size, self.p_rect, device)
        m_ir = self._random_mask(n, h, w, self.num_holes, self.hole_size, self.p_rect, device)

        overlap = (m_vi == 0) & (m_ir == 0)
        if overlap.any():
            chooser = torch.rand_like(m_vi).round()  # 0/1
            m_ir = torch.where(overlap & (chooser == 0), torch.ones_like(m_ir), m_ir)
            m_vi = torch.where(overlap & (chooser == 1), torch.ones_like(m_vi), m_vi)

        return m_vi, m_ir

    def random(self, vi):
        n, _, h, w = vi.shape
        return self._random_mask(n, h, w, self.num_holes, self.hole_size, self.p_rect, vi.device)

    def modal_absence(self, vi, which='vis'):
        n, _, h, w = vi.shape
        m = torch.ones(n, 1, h, w, device=vi.device)
        if which == 'vis':
            return m * 0., torch.ones_like(m)   # VIS=0, IR=1
        else:
            return torch.ones_like(m), m * 0.   # VIS=1, IR=0



save_model_path=r'./'

model=Network().cuda()

masker = InterventionMasker()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
fusion_loss = fusion_loss_vif().cuda()
loss_inv_fn = CausalInvarianceLoss()
loss_necess_fn = CausalNecessityLoss()
SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')

dataset = FusionDataset2(train=True)
train_loader = data.DataLoader(dataset, batch_size=16,num_workers=8,shuffle=True, drop_last=True)


for epoch in range(50):
    model.train()
    for ir, vi in train_loader:
        vi = vi.cuda()
        ir = ir.cuda()

        out, _, gates_out = model(vi, ir)

        # (1) complementary masks
        m_vi_c, m_ir_c = masker.complementary(vi, ir)
        y_comp, _, gates_comp = model(vi * m_vi_c, ir * m_ir_c)

        # (2) random mask
        m_r = masker.random(vi)
        y_rand, _, gates_rand = model(vi * m_r, ir * m_r)

        # (3) modality dropout
        mv0, mi1 = masker.modal_absence(vi, which='vis')
        y_abs_vis0, _, _ = model(vi * mv0, ir * mi1)

        mv1, mi0 = masker.modal_absence(vi, which='ir')
        y_abs_ir0, _, _ = model(vi * mv1, ir * mi0)

        def _agg_gate(gdict, target_hw):
            g3 = F.interpolate(gdict['g3'], size=target_hw, mode='bilinear', align_corners=False)
            g4 = F.interpolate(gdict['g4'], size=target_hw, mode='bilinear', align_corners=False)
            g5 = F.interpolate(gdict['g5'], size=target_hw, mode='bilinear', align_corners=False)
            return (g3 + g4 + g5) / 3.0

        target_hw = out.shape[-2:]
        g_base = _agg_gate(gates_out, target_hw)
        g_comp = _agg_gate(gates_comp, target_hw)
        g_rand = _agg_gate(gates_rand, target_hw)

        loss_inv = loss_inv_fn(
            y_base=out,
            y_do_list=[y_comp, y_rand],
            g_list=[g_comp, g_rand]
        )

        loss_necess = loss_necess_fn(out, y_abs_vis0, y_abs_ir0)

        loss_f = fusion_loss(ir, vi, out)

        alpha, beta = 0.1, 0.05
        loss = loss_f + alpha * loss_inv + beta * loss_necess

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f'Epoch: {epoch}  Loss_f: {loss_f.item():.6f}  Loss_inv: {loss_inv.item():.6f}  Loss_nec: {loss_necess.item():.6f}')

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'{(epoch + 1)}_model.pth')
