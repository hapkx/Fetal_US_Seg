"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
from visdom import Visdom
viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
# from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.mydataloader import MyDataset
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
import torchvision.transforms.functional as F

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

class JointCompose:
    """支持同时处理图像和掩码的 Compose 类"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)  # 确保每个变换都返回 (img, mask)
        return img, mask

class JointToTensor:
    def __call__(self, img, mask):  # 正确：接收两个参数
        return F.to_tensor(img), F.to_tensor(mask)  # 正确：返回两个结果

class JointRandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img, mask):
        # 获取图像尺寸（PIL图像为宽高，转换为高宽）
        width, height = img.size
        target_h , target_w = self.output_size  # 目标高宽

        # 确保裁剪尺寸不超过原图
        if height < target_h  or width < target_w:
            raise ValueError(f"裁剪尺寸{self.output_size}大于原图尺寸{(height, width)}")

        # 生成随机左上角坐标
        top = th.randint(0, height - target_h  + 1, size=(1,)).item()
        left = th.randint(0, width - target_w  + 1, size=(1,)).item()

        # 应用相同参数裁剪图像和掩码
        img_cropped = F.crop(img, top, left, target_h , target_w )
        mask_cropped = F.crop(mask, top, left, target_h , target_w )

        return img_cropped, mask_cropped
    


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    print("creating data loader")
    # 数据加载器设置
    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    else:
        transform_train = JointCompose([
            JointRandomCrop(output_size=(256, 256)),  # 随机裁剪为256x256
            JointToTensor()
        ])
        print("Your current directory : ",args.data_dir)
        ds = MyDataset(args, args.data_dir, transform_train, mode="test")

    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 加载预训练模型
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")     # 从指定路径加载预训练模型的状态字典。
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            # 如果状态字典中的键包含 module.，则去除该前缀
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    # 将状态字典加载到模型中
    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    for _ in range(len(data)):
        b, m, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :3, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
       
        print("*****************")
        # print(path[0])
        slice_ID = path[0].split('.')[0].split('/')[-1]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        # 对于每个样本，生成 args.num_ensemble 个分割掩码
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            # sample, x_noisy, org, cal, cal_out = sample_fn(
            #     model,
            #     (args.batch_size, 3, args.image_size, args.image_size), img,
            #     step = args.diffusion_steps,
            #     clip_denoised=args.clip_denoised,
            #     model_kwargs=model_kwargs,
            # )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            enslist.append(sample)

            # co = th.tensor(cal_out)
            # enslist.append(co)
            
            # s = th.tensor(sample)
            # viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            # th.save(s, args.out_dir + str(slice_ID)+'_output'+str(i)+".jpg") #save the generated mask
            
            # vutils.save_image(s, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)


        ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)




def create_argparser():
    defaults = dict(
        data_name = 'mydata',
        data_dir="./data/mydata",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/0421/unet4',
        multi_gpu = None, #"0,1,2"
        debug = False,
        model_path='/home/nas2/biod/piankexin/FetalSeg/result/0421/unet4/emasavedmodel_0.9999_100000.pt',
        image_size=256,
        num_channels=64,
        class_cond=False,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma= True,
        use_scale_shift_norm= False,
        attention_resolutions=16,
        diffusion_steps=1000,
        noise_schedule='linear',
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
