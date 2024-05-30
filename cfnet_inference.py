import torch
import numpy as np
import cv2
from copy import deepcopy
from cfnet_arch import ChromaFusionRRDBNet, rgb2lab, lab2rgb
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize, Resize
from PIL import Image
from rich import print

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


class CFNetModel:
    def __init__(self, load_path='experiments/ChromaFusionNet_ImageNet_bs16_rrdb/models/net_g_800000.pth', device='cuda:0'):
        self.device = device
        self.model = self.load_cfnet(load_path)
        self.fusion_transform = self.fusion_transform = Compose([Resize((224, 224)),ToTensor()])

    def load_cfnet(self, load_path):
        model = ChromaFusionRRDBNet()
        model.load_state_dict(torch.load(load_path, map_location='cpu')['params'], strict=True)
        model.eval()
        model = model.to(self.device)
        return model

    def resize_ab(self, img_ori, img_palette):
        img_palette = np.array(img_palette, dtype=np.uint8) if type(img_palette) != np.ndarray else img_palette
        img_ori = np.array(img_ori, dtype=np.uint8) if type(img_ori) != np.ndarray else img_ori

        img_palette = cv2.cvtColor(img_palette, cv2.COLOR_RGB2LAB)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2LAB)

        img_palette = cv2.resize(img_palette, (img_ori.shape[1], img_ori.shape[0]))[:, :, 1:]
        img_lab = np.concatenate((img_ori[:, :, :1], img_palette), axis=-1)
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return img_rgb

    @torch.no_grad()
    def feed_data(self, img, mask_edge):
        images = self.fusion_transform(img.convert('RGB')).unsqueeze(0).to(self.device)
        mask_edge = self.fusion_transform(mask_edge.convert('L')).unsqueeze(0).to(self.device)

        imgs_lab = rgb2lab(images) 
        imgs_abmaskout = deepcopy(imgs_lab)
        mask_cond = mask_edge.repeat(1, 2, 1, 1) > 1e-6
        imgs_abmaskout[:, 1:, :, :][mask_cond] = 0

        return imgs_abmaskout, mask_edge, imgs_lab

    def forward_fusion(self, imgs_abmaskout, mask_edge, imgs_lab):
        x = torch.cat([imgs_abmaskout, mask_edge], dim=1)
        pred = self.model(x)
        if type(pred) == dict:
            pred = pred['rrdb_output']
        pred = torch.cat([imgs_lab[:, :1, :, :], pred], dim=1)
        return pred

    def postprocess_fusion(self, pred_imgs):
        pred_imgs = lab2rgb(pred_imgs)  # Placeholder for actual lab2rgb conversion
        pred_imgs = tensor2img(pred_imgs, rgb2bgr=False)  # Placeholder for tensor2img conversion
        return pred_imgs  

    def inference(self, ori_img, palette_img, mask_edge):
        if type(ori_img) == np.ndarray:
            ori_img = Image.fromarray(ori_img)
        if type(palette_img) == np.ndarray:
            palette_img = Image.fromarray(palette_img)
        if type(mask_edge) == np.ndarray:
            mask_edge = Image.fromarray(mask_edge)

        imgs_abmaskout, mask_edge, imgs_lab = self.feed_data(palette_img, mask_edge)
        pred_imgs = self.forward_fusion(imgs_abmaskout, mask_edge, imgs_lab)
        pred_imgs = self.postprocess_fusion(pred_imgs)
        pred_imgs = self.resize_ab(ori_img, pred_imgs)
        return pred_imgs

if __name__ == '__main__':
    cfnet_model = CFNetModel('experiments/ChromaFusionNet_ImageNet_bs16_rrdb/models/net_g_800000.pth')
    ori_img = cv2.cvtColor(cv2.imread('images/ori.jpeg'), cv2.COLOR_BGR2RGB)
    palette_img = cv2.cvtColor(cv2.imread('images/color.jpeg'), cv2.COLOR_BGR2RGB)
    mask_edge = cv2.imread('images/mask_edge.jpeg', 0)

    pred_imgs = cfnet_model.inference(ori_img, palette_img, mask_edge)
    cv2.imwrite('images/cfnet_output.jpg', cv2.cvtColor(pred_imgs, cv2.COLOR_RGB2BGR))
    print('Output saved as images/cfnet_output.jpg')
