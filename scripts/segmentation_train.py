"""
Train a diffusion model on images.
"""
import sys
import argparse
# sys.path.append("..")
sys.path.append(".")
from PIL import Image
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.mydataloader import MyDataset
import os
from torchinfo import summary

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torchvision.transforms.functional as F
import torch as th
from guided_diffusion.train_util import TrainLoop
import torchvision.transforms as transforms
import os
# from visdom import Visdom
# viz = Visdom(port=8850)

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
    logger.configure(dir=args.out_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(model)

    # 
    # input_size = (1, 3, 256, 256)  # 输入的尺寸，根据实际情况修改
    # summary(model, input_size=input_size)
    # 

    # print(model)
    # 保存模型结构到文本文件
    # with open('model_structure.txt', 'w') as f:
    #     # 将模型结构输出重定向到文件
    #     print(model, file=f)

    # print("模型结构已保存到 model_structure.txt")

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")

    transform_train = JointCompose([
        JointRandomCrop(output_size=(256, 256)),  # 随机裁剪为256x256
        JointToTensor()
    ])
    ds = MyDataset(args, args.data_dir, transform=transform_train)
    print("Your current directory : ",args.data_dir)
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    img, mask, name = ds[0]
    # print(img.shape)      # torch.Size([3, 256, 256])
    # print(mask.shape)     # torch.Size([1, 256, 256, 3])
    # print(name)

    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)

    logger.log("training...")

    start.record()

    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    end.record()
    # 等待事件完成（确保同步）
    th.cuda.synchronize()
    time_elapsed = start.elapsed_time(end)
    print(f"GPU操作耗时: {time_elapsed} ms")


def create_argparser():
    defaults = dict(
        data_name = 'mydata',
        data_dir="./data/mydata",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=100000,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        out_dir='./result/0421/unet1',
        image_size=256,
        num_channels=64,
        class_cond=False,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=False,
        batch_size=1,
        diffusion_steps=1000,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="2,4,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        dpm_solver = False,
        use_new_attention_order=False,
        noise_schedule="linear",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
