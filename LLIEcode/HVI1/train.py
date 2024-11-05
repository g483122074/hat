import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm


opt = option().parse_args()

# 保证每次运行相同的代码时，生成的随机数序列一致，从而使实验结果具有可重复性和可靠性。
def seed_torch():
    # 生成一个在1到1000000之间的随机整数"seed",这个种子用于初始化 随机数生成器
    seed = random.randint(1, 1000000)
    # 设置python内置"random"模块的 随机数生成器 的种子
    random.seed(seed)
    # 设置 NumPy的 随机数生成器 的种子
    np.random.seed(seed)
    # 设置 PyTorch 的 CPU 随机数生成器的种子
    torch.manual_seed(seed)
    # 设置 PyTorch 的单个 GPU 随机数生成器的种子
    torch.cuda.manual_seed(seed)
    # 设置 PyTorch 所有 GPU 的随机数生成器的种子（如果使用多GPU）
    torch.cuda.manual_seed_all(seed)
    # 设置环境变量 "PYTHONHASHSEED"的值为 seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()

    # 当 cudnn.benchmark 设置为 True 时，cuDNN 会尝试不同的卷积算法，并选择最优的一个。
    # 这对于卷积神经网络的训练速度优化有帮助，但只有在输入大小不变的情况下才有效。
    # cuDNN 是NVIDIA 提供的一套高性能 GPU 加速库，专门用于加速深度神经网络的计算
    cudnn.benchmark = True

    # 设置环境变量 CUDA_VISIBLE_DEVICES，这个环境变量控制哪个 GPU 对当前进程可见
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # 获取命令行参数中的"gpu_mode" 的值，决定是否使用GPU
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()  # 将模型设置为训练模式
    loss_print = 0  # 用于打印累计损失
    pic_cnt = 0     # 累计处理的图像数量
    loss_last_10 = 0  # 最近10个batch的累计损失
    pic_last_10 = 0   # 最近10个batch的累计图像数量

    train_len = len(training_data_loader)   # 训练数据的总批次数
    iter = 0   # 当前迭代次数

    torch.autograd.set_detect_anomaly(True)    # 启用异常检测，帮助调试

    for batch in tqdm(training_data_loader):  # 迭代加载训练数据
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        output_rgb = model(im1)  # 前向传播，获取模型输出
        gt_rgb = im2             # 设置真实图像

        # 获取low和high的 hvi特征
        output_hvi = model.RGB_to_THSV(output_rgb)  
        gt_hvi = model.RGB_to_THSV(gt_rgb)

        # 计算损失
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        iter += 1
        
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        
        #  更新损失和图像计数
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        pic_cnt += 1
        pic_last_10 += 1

        # 每个 epoch 的最后一个 batch
        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_last_10/pic_last_10, optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0

            # 保存输出图像和真实图像
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')

    return loss_print, pic_cnt   # 返回累计损失和图像数量
                

# 用于保存训练权重
def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path


# 获取训练数据和验证数据
def load_datasets():
    print('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad:
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lol_blur:
            train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lolv2_real:
            train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        if opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader


# 构建模型
def build_model():
    print('===> Building model ')
    model = CIDNet().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

# 创建优化器
def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

# 初始化损失
def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True

            # LOL three subsets
            if opt.lol_v1:
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.lolv2_real:
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.lolv2_syn:
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.lol_blur:
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.SID:
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.SICE_grad:
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False

            im_dir = opt.val_folder + output_folder + '*.png'
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
        torch.cuda.empty_cache()
    
    with open("./results/training/metrics.md", "w") as f:
        f.write("dataset: "+ output_folder + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")  
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        for i in range(len(psnr)):
            f.write(f"| {(i+1)*10} | { psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")  
        