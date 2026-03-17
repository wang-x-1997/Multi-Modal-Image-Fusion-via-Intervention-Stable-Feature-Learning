import math
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

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

import glob
def prepare_data(dataset):
    # data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data_dir =dataset
    data = glob.glob(os.path.join(data_dir, "IR*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "IR*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "IR*.bmp")))
    a = data[0][len(str(data_dir))+1:-6]
    data.sort(key=lambda x:int(x[len(str(data_dir))+2:-4]))
    return data
def prepare_data1(dataset):
    # data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data_dir = dataset
    data = glob.glob(os.path.join(data_dir, "VIS*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "VIS*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "VIS*.bmp")))
    data.sort(key=lambda x:int(x[len(str(data_dir))+3:-4]))
    return data
import cv2

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


# ======================================
# Causal Feature Integrator (CFI)
# ======================================

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

        # 1x1投影生成Q,K,V
        self.q_v = nn.Conv2d(dim, dim, 1)
        self.k_v = nn.Conv2d(dim, dim, 1)
        self.v_v = nn.Conv2d(dim, dim, 1)

        self.q_i = nn.Conv2d(dim, dim, 1)
        self.k_i = nn.Conv2d(dim, dim, 1)
        self.v_i = nn.Conv2d(dim, dim, 1)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim//2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 融合后再refine
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
        # Q,K,V
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

        # ===== level 3 ( H/4) =====
        # x3 = torch.cat([v3, i3], dim=1)            # [N,2dim,H/4,W/4]
        # x3 = self.adjust_channels3(x3)             # [N,dim,H/4,W/4]
        f3, g3 = self.cfi3(v3, i3)                 # f3:[N,dim,H/4,W/4]
        x3 =  f3

        # ===== level 4 ( H/2) =====
        x4_up = F.interpolate(x3, size=v2.shape[-2:], mode='bilinear', align_corners=False)
        f4, g4 = self.cfi4(v2, i2)
        x4 = x4_up  + f4   # [N,3dim,H/2,W/2]
        x4 = self.adjust_channels4(x4)             # [N,dim,H/2,W/2]

        # ===== level 5 ( H) =====
        x5_up = F.interpolate(x4, size=v1.shape[-2:], mode='bilinear', align_corners=False)
        f5, g5 = self.cfi5(v1, i1)
        x5 = x5_up + f5   # [N,3dim,H,W]
        x5 = self.adjust_channels5(x5)             # [N,dim,H,W]

        out = (self.final_decoder(x5) )            # [N,1,H,W] in [0,1]

        intermediate_outputs = (x3, x4, x5)
        gates = {'g3': g3, 'g4': g4, 'g5': g5}

        return out, intermediate_outputs, gates


model=Network().cuda()

model.load_state_dict(torch.load(r"./best.pth"))
model.eval()

for n in  ['FLIR']:
    ir = prepare_data(r"D:\Image_Data\IRVI\AUIF Datasets\16x\Test_{}/".format(n))
    vi = prepare_data1(r"D:\Image_Data\IRVI\AUIF Datasets\16x\Test_{}/".format(n))
    save_path = r'./Fused/{}/'.format(n)

    # names_ir = os.listdir(ir)
    # names_vi = os.listdir(vi)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(ir)):
        print( ir[i])
        # print(names_vi[i])
        img_ir = cv2.imread(ir[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        img_vi = cv2.imread(vi[i], cv2.IMREAD_COLOR).astype(np.float32)
        img_vi = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
        Y = img_vi[:, :, 0].astype(np.float32) / 255.

        img_ir = np.expand_dims(img_ir, axis=-1)
        Y = np.expand_dims(Y, axis=-1)

        img_ir = torch.from_numpy(img_ir).float().permute(2, 0, 1).unsqueeze(0).cuda()
        Y = torch.from_numpy(Y).float().permute(2, 0, 1).unsqueeze(0).cuda()

        with torch.no_grad():
            output, _, g = model(Y, img_ir)
        # output = g['g5']
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.expand_dims(output, axis=-1)
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        output = np.concatenate([output, img_vi[:, :, 1:]], axis=-1)
        output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(save_path + str(i+1)+'.jpg', output)