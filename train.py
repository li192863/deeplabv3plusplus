import argparse
import time
from datetime import datetime
from functools import partial

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from dataset import WSISegmentationDataset
from presets import SegmentationPresetTrain, SegmentationPresetEval
from model import MyNet
from utils import init_file_logger, make_directory

# 默认配置
DEFAULT_RUNNING_DIR = './runs/' + datetime.now().strftime("%m%d-%H%M")
DATASET_ROOT_PATH = '../../autodl-tmp/Segmentation'
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 8
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH_ACC = f'{DEFAULT_RUNNING_DIR}/model_acc.pth'
DEFAULT_SAVE_PATH_DICE = f'{DEFAULT_RUNNING_DIR}/model_dice.pth'
DEFAULT_SAVE_PATH_IOU = f'{DEFAULT_RUNNING_DIR}/model_iou.pth'
DEFAULT_LAST_SAVE_PATH = f'{DEFAULT_RUNNING_DIR}/model_last.pth'
DEFAULT_WORKERS = 8
# 初始化
classes = ['_background_', 'cancerous', 'paracancerous']


def get_dataloader(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    # 使用数据集
    train_data = WSISegmentationDataset(DATASET_ROOT_PATH,
                                        type='train',
                                        transforms=SegmentationPresetTrain())
    test_data = WSISegmentationDataset(DATASET_ROOT_PATH,
                                       type='val',
                                       transforms=SegmentationPresetEval())

    # 定义数据加载器
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.workers)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=opt.batch_size,
                                                  shuffle=False,
                                                  num_workers=opt.workers)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer, opt):
    """
    训练模型
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param opt:
    :return:
    """
    total_loss = 0.0
    model.train()  # Sets the module in training mode
    with tqdm(dataloader,
              desc=f'Epoch {opt.epoch}/{opt.epochs}, train',
              total=len(dataloader)) as pbar:  # 进度条
        for X, y in pbar:
            # 前向传播
            X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
            pred = model(X)  # 预测结果
            loss = loss_fn(pred, y)  # 计算损失

            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            # 打印信息
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{total_loss:>7f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:>7f}'
            })
    return total_loss  # 返回训练损失


def test(dataloader, model, loss_fn, opt, num_classes):
    """
    测试模型
    :param dataloader:
    :param model:
    :param loss_fn:
    :param opt:
    :return:
    """
    num_batches = len(dataloader)
    loss, confmat = 0, utils.ConfusionMatrix(num_classes)
    model.eval()  # Sets the module in evaluation mode
    with torch.no_grad():  # Disabling gradient calculation
        with tqdm(dataloader,
                  desc=' ' * (len(str(opt.epoch)) + len(str(opt.epochs)) + 9) +
                  'test',
                  total=len(dataloader)) as pbar:  # 进度条
            for X, y in pbar:
                X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
                pred = model(X)  # 预测结果
                loss += loss_fn(pred, y).item()  # 计算损失
                confmat.update(y.flatten(), pred.argmax(1).flatten())  # 更新指标
                pbar.set_postfix({'Avg loss': f'{loss / num_batches:>8f}'})
    return confmat  # 返回准确率


def show_time_elapse(start, end=None, prefix='', suffix=''):
    """
    显示运行时间
    :param start:
    :param end:
    :param prefix:
    :param suffix:
    :return:
    """
    end = end or time.time()
    time_elapsed = end - start  # 单位为秒
    hours = time_elapsed // 3600  # 时
    minutes = (time_elapsed - hours * 3600) // 60  # 分
    seconds = (time_elapsed - hours * 3600 - minutes * 60) // 1  # 秒
    if hours == 0:  # 0 hours x minutes x seconds
        if minutes == 0:  # 0 hours 0 minutes x seconds
            print(prefix + f' {seconds:.0f}s ' + suffix)
        else:  # 0 hours x minutes x seconds
            print(prefix + f' {minutes:.0f}m {seconds:.0f}s ' + suffix)
    else:  # x hours x minutes x seconds
        print(prefix + f' {hours:.0f}h {minutes:.0f}m {seconds:.0f}s ' +
              suffix)


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size',
                        type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help='batch size')
    parser.add_argument('--device',
                        default=DEFAULT_DEVICE,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-path-acc',
                        default=DEFAULT_SAVE_PATH_ACC,
                        help='model save path for acc')
    parser.add_argument('--save-path-dice',
                        default=DEFAULT_SAVE_PATH_DICE,
                        help='model save path for dice')
    parser.add_argument('--save-path-iou',
                        default=DEFAULT_SAVE_PATH_IOU,
                        help='model save path for iou')
    parser.add_argument('--last-save-path',
                        default=DEFAULT_LAST_SAVE_PATH,
                        help='model last save path')
    parser.add_argument('--running-dir',
                        default=DEFAULT_RUNNING_DIR,
                        help='running-dir')
    parser.add_argument('--workers',
                        default=DEFAULT_WORKERS,
                        help='max dataloader workers')
    return parser.parse_args()


def main(opt):
    """
    主函数
    :param opt:
    :return:
    """
    # 计时
    start = time.time()
    # 设备
    logger.info(f'Using {opt.device} device')
    # 数据
    train_dataloader, test_dataloader = get_dataloader(opt)
    # 模型
    num_classes = len(classes)
    model = MyNet(bilinear=False, num_classes=num_classes).to(opt.device)
    # model.load_state_dict(torch.load(opt.save_path)) if os.path.exists(opt.save_path) else ''
    # 参数
    # params = [
    #     {'params': [p for p in model.encoder.parameters() if p.requires_grad]},
    #     {'params': [p for p in model.classifier.parameters() if p.requires_grad]}
    # ]
    loss_fn = partial(nn.functional.cross_entropy, ignore_index=255)  # 损失函数
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)  # 优化器
    optimizer = torch.optim.Adam(model.parameters())  # 优化器
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=len(train_dataloader) * opt.epochs, power=0.9
    )  # Decays the learning rate of each parameter group using a polynomial function in the given total_iters.
    # 训练
    best_gAcc, best_mDice, best_mIou = 0.0, 0.0, 0.0
    best_gAcc_epoch, best_mDice_epoch, best_mIou_epoch = 0, 0, 0
    writer = SummaryWriter(opt.running_dir)
    for epoch in range(opt.epochs):
        opt.epoch = epoch + 1  # 设置当前循环轮次
        logger.info(f'Epoch {opt.epoch}/{opt.epochs}:')
        loss = train(train_dataloader, model, loss_fn, optimizer, opt)  # 训练
        lr_scheduler.step()  # 更新学习率
        confmat = test(test_dataloader, model, loss_fn, opt, num_classes)  # 测试
        logger.info(confmat)
        acc, iu, dice, gAcc, mIou, mDice = confmat.compute()
        if gAcc > best_gAcc:
            best_gAcc_epoch = opt.epoch
            best_gAcc = max(gAcc, best_gAcc)
            # 保存
            torch.save(model.state_dict(), opt.save_path_acc)
            logger.info(
                f'Saved PyTorch Model State to {opt.save_path_acc}, model\'s best gAcc is {100 * best_gAcc:>0.1f}%'
            )
        if mDice > best_mDice:
            best_mDice_epoch = opt.epoch
            best_mDice = max(mDice, best_mDice)
            # 保存
            torch.save(model.state_dict(), opt.save_path_dice)
            logger.info(
                f'Saved PyTorch Model State to {opt.save_path_dice}, model\'s best mDice is {100 * best_mDice:>0.1f}%'
            )
        if mIou > best_mIou:
            best_mIou_epoch = opt.epoch
            best_mIou = max(mIou, best_mIou)
            # 保存
            torch.save(model.state_dict(), opt.save_path_iou)
            logger.info(
                f'Saved PyTorch Model State to {opt.save_path_iou}, model\'s best mIOU is {100 * best_mIou:>0.1f}%'
            )
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('gAcc', gAcc, epoch)
        writer.add_scalar('mDice', mDice, epoch)
        writer.add_scalar('mIOU', mIou, epoch)
        writer.add_scalar('learning rate',
                          lr_scheduler.get_last_lr()[0], epoch)
        print('\n')
    writer.close()
    logger.info(f'Done!')
    # 保存
    torch.save(model.state_dict(), opt.last_save_path)
    logger.info(f'Saved PyTorch Model State to {opt.last_save_path}')
    logger.info(
        f'model\'s best gAcc is {100 * best_gAcc:>0.2f}%(epoch{best_gAcc_epoch})'
    )
    logger.info(
        f'model\'s best mDice is {100 * best_mDice:>0.2f}%(epoch{best_mDice_epoch})'
    )
    logger.info(
        f'model\'s best mIou is {100 * best_mIou:>0.2f}%(epoch{best_mIou_epoch})'
    )

    # 计时
    show_time_elapse(start, time.time(), 'Training complete in')
    return best_mDice


if __name__ == '__main__':
    opt = parse_opt()
    # 目录
    make_directory(opt.running_dir)
    # 记录
    logger = init_file_logger(file=f'{opt.running_dir}/report.txt',
                              name='my-logger')
    best_mDice = main(opt)
