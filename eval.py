import argparse
import colorsys
from functools import partial

import torch
import torchvision.transforms.functional as F
from PIL import ImageFont, ImageDraw, Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks, make_grid
from tqdm import tqdm

import utils
from dataset import WSISegmentationDataset
from model import MyNet
from presets import SegmentationPresetEval

# 默认配置
from utils import make_directory, init_file_logger

DATETIME= '02151057'
DEFAULT_RUNNING_DIR = './runs/' + DATETIME[:4] + '-' + DATETIME[4:]
DATASET_ROOT_PATH = '../../autodl-tmp/Segmentation'
# DEFAULT_MODEL_PATH = f'{DEFAULT_RUNNING_DIR}/model_dice.pth'
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEFAULT_SAVE_PATH_ACC = f'{DEFAULT_RUNNING_DIR}/model_acc.pth'
# DEFAULT_SAVE_PATH_DICE = f'{DEFAULT_RUNNING_DIR}/model_dice.pth'
# DEFAULT_SAVE_PATH_IOU = f'{DEFAULT_RUNNING_DIR}/model_iou.pth'
DEFAULT_WORKERS = 8
classes = ['_background_', 'cancerous', 'paracancerous']


def get_test_data(opt, sample_list=None):
    """
    获取测试数据
    :return:
    """
    # 使用数据集
    val_data = WSISegmentationDataset(DATASET_ROOT_PATH, type='val', transforms=SegmentationPresetEval())
    test_data = WSISegmentationDataset(DATASET_ROOT_PATH, type='test', transforms=SegmentationPresetEval())

    # 定义数据加载器
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    if sample_list is None:
        sample_list = torch.randint(0, len(val_dataloader), (opt.batch_size,))
    x, y = list(zip(*[val_data[index] for index in sample_list]))
    x, y = torch.stack(x, dim=0).to(opt.device), torch.stack(y, dim=0).to(opt.device)
    return val_dataloader, test_dataloader, x, y


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
        with tqdm(dataloader, desc='test', total=len(dataloader)) as pbar:  # 进度条
            for X, y in pbar:
                X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
                pred = model(X)  # 预测结果
                loss += loss_fn(pred, y).item()  # 计算损失
                confmat.update(y.flatten(), pred.argmax(1).flatten())  # 更新指标
                pbar.set_postfix({'Avg loss': f'{loss / num_batches:>8f}'})
    return confmat  # 返回准确率


def show_sematic_segmentation_result(images, labels, masks, image_size=None, colors=None, count_background=False, save_path='result.png'):
    """
    展示语义分割结果
    @param images:
    @param labels:
    @param masks:
    @param image_size:
    @param colors:
    @param count_background:
    @param save_path:
    @return:
    """
    # 预处理图片
    num_classes = len(colors)
    # 是否填背景
    if count_background:
        labels_bool = torch.stack([label == i for label in labels for i in range(num_classes)], dim=0) \
            .cpu().reshape(-1, num_classes, labels.shape[-2], labels.shape[-1])
        masks_bool = torch.stack([mask == i for mask in masks for i in range(num_classes)]) \
            .cpu().reshape(-1, num_classes, labels.shape[-2], labels.shape[-1])
    else:
        labels_bool = torch.stack([label == i for label in labels for i in range(1, num_classes + 1)]) \
            .cpu().reshape(-1, num_classes, labels.shape[-2], labels.shape[-1])
        masks_bool = torch.stack([mask == i for mask in masks for i in range(1, num_classes + 1)]) \
            .cpu().reshape(-1, num_classes, labels.shape[-2], labels.shape[-1])
    # 恢复原始图片
    mean = torch.tensor([0.723, 0.485, 0.608]).reshape((3, 1, 1))  # 预训练时标准化的均值
    std = torch.tensor([0.293, 0.377, 0.333]).reshape((3, 1, 1))  # 预训练时标准化的方差
    images = [torch.as_tensor(torch.clip((image.cpu() * std + mean) * 255, 0, 255), dtype=torch.uint8) for image in images]  # 对输入tensor进行处理
    images = [remove_edge(image) for image in images]

    # 绘制每张图
    num_images = len(images)
    image_size = image_size or (sum(list(map(lambda x: x.shape[2], images))) // num_images,
                                sum(list(map(lambda x: x.shape[1], images))) // num_images)  # 获取图片大小(W, H)
    origins = [get_image_tensor(image, mask, image_size, colors, title='原始', alpha=0.0) for image, mask in zip(images, labels_bool)]
    targets = [get_image_tensor(image, mask, image_size, colors, title='标签') for image, mask in zip(images, labels_bool)]
    preds = [get_image_tensor(image, mask, image_size, colors, title='预测') for image, mask in zip(images, masks_bool)]
    origins.extend(targets)
    origins.extend(preds)
    # 生成网格图
    result = make_grid(origins, nrow=num_images)
    result = F.to_pil_image(result)
    result.save(save_path)
    result.show()


def get_image_tensor(image, mask, image_size, colors, title='', alpha=0.8):
    """
    获取单张图片
    @param image: 图片tensor
    @param mask: 蒙版tensor 值为bool
    @param image_size: 图片尺寸
    @param colors: 颜色
    @param title: 标题
    @return:
    """
    font = ImageFont.truetype(font='data/Microsoft YaHei.ttf', size=10 * 3)  # 设置字体
    image = draw_segmentation_masks(image, mask, colors=colors, alpha=alpha)
    image = F.to_pil_image(image)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), title, font=font, fill=0)
    image = letterbox_image(image, image_size)
    image = F.to_tensor(image)
    return image


def remove_edge(image):
    """
    去除图片右边以及下边外围多余黑边
    :param image:
    :return:
    """
    # 确保图片右下角像素为黑边
    if torch.sum(image[:, -1, -1] != torch.tensor([255, 255, 255])).item() != 0:
        return image
    mask = image == image[:, -1, -1].reshape(3, 1, 1)  # equals last pixel mask
    mask = torch.all(mask, dim=0)  # 深度方向逻辑与操作，得到[H, W]的张量，相当于mask[0, ...] & mask[1, ...] & mask[2, ...]
    line_h, line_w = torch.all(mask, dim=1), torch.all(mask, dim=0) # 高度方向逻辑与操作
    idx_h, idx_w = torch.nonzero(line_h).squeeze(1), torch.nonzero(line_w).squeeze(1)  # 获取值为true的下标
    max_h = idx_h[0] if len(idx_h) != 0 else image.size(1)  # 获取图片边界对应的下标max_h
    max_w = idx_w[0] if len(idx_w) != 0 else image.size(2)  # 获取图片边界对应的下标max_w
    image = image[:, :max_h, :max_w]
    return image


def letterbox_image(image, image_size):
    """
    图片等比例缩放
    :param image: PIL image
    :param image_shape: (W, H)
    :return:
    """
    # 获取原始宽高和需要的宽高
    old_width, old_height = image.size
    new_width, new_height = image_size
    # 缩放图片有效区域
    scale = min(new_width / old_width, new_height / old_height)  # 图片有效区域缩放比例
    valid_width, valid_height = int(old_width * scale), int(old_height * scale)
    image = image.resize((valid_width, valid_height))
    # 填充图片无效区域
    origin = [(new_width - valid_width) // 2, (new_height - valid_height) // 2]
    result = Image.new(mode=image.mode, size=(new_width, new_height), color=(128, 128, 128))
    result.paste(image, origin)
    return result


def visualize(model, x, y, num_classes, save_path, opt):
    """
    可视化模型效果
    @param model:
    @param x:
    @param y:
    @param num_classes:
    @param save_path:
    @param opt:
    @return:
    """
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        pred = model(x)
        # 处理结果数据
        masks = pred.argmax(1).to('cpu')
        # 标签数字转化为标签名称
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]  # 色调 饱和度1 亮度1
        color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color))
        show_sematic_segmentation_result(x, y, masks, image_size=[224, 224], colors=colors, save_path=f'{opt.running_dir}/{save_path}')


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--running-dir', default=DEFAULT_RUNNING_DIR, help='running-dir')
    parser.add_argument('--model-path', default=f'{parser.parse_args().running_dir}/model_dice.pth', help='model weights path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-path-acc', default=f'{parser.parse_args().running_dir}/model_acc.pth', help='model save path for acc')
    parser.add_argument('--save-path-dice', default=f'{parser.parse_args().running_dir}/model_dice.pth', help='model save path for dice')
    parser.add_argument('--save-path-iou', default=f'{parser.parse_args().running_dir}/model_iou.pth', help='model save path for iou')
    parser.add_argument('--workers', default=DEFAULT_WORKERS, help='max dataloader workers')
    return parser.parse_args()


def main(opt):
    # # 设备
    # logger.info(f'Using {opt.device} device')
    # # 数据
    # val_dataloader, test_dataloader, x, y = get_test_data(opt, [0, 15, 26, 28])
    # # 模型
    # num_classes = len(classes)
    # model = MyNet(bilinear=False, num_classes=num_classes).to(opt.device)
    # # 损失
    # loss_fn = partial(nn.functional.cross_entropy, ignore_index=255)  # 损失函数
    # # 评估
    # logger.info('Model with best gAcc:')
    # model.load_state_dict(torch.load(opt.save_path_acc))
    # logger.info(test(val_dataloader, model, loss_fn, opt, num_classes))
    # # logger.info(test(test_dataloader, model, loss_fn, opt, num_classes))
    # visualize(model, x, y, num_classes, 'result_gAcc.png', opt)
    # logger.info('Model with best mDice:')
    # model.load_state_dict(torch.load(opt.save_path_dice))
    # logger.info(test(val_dataloader, model, loss_fn, opt, num_classes))
    # # logger.info(test(test_dataloader, model, loss_fn, opt, num_classes))
    # visualize(model, x, y, num_classes, 'result_mDice.png', opt)
    # logger.info('Model with best mIou:')
    # model.load_state_dict(torch.load(opt.save_path_iou))
    # logger.info(test(val_dataloader, model, loss_fn, opt, num_classes))
    # # logger.info(test(test_dataloader, model, loss_fn, opt, num_classes))
    # visualize(model, x, y, num_classes, 'result_mIou.png', opt)

    # 数据
    val_dataloader, test_dataloader, x, y = get_test_data(opt, [8])  # 12, 26, 31, 40, 43
    # 模型
    num_classes = len(classes)
    model = MyNet(bilinear=False, num_classes=num_classes).to(opt.device)
    # 损失
    loss_fn = partial(nn.functional.cross_entropy, ignore_index=255)  # 损失函数
    # 评估
    model.load_state_dict(torch.load(opt.save_path_acc))
    # logger.info(test(val_dataloader, model, loss_fn, opt, num_classes))
    # logger.info(test(test_dataloader, model, loss_fn, opt, num_classes))
    visualize(model, x, y, num_classes, 'result_gAcc.png', opt)
    # logger.info('Model with best mDice:')
    model.load_state_dict(torch.load(opt.save_path_dice))
    # logger.info(test(val_dataloader, model, loss_fn, opt, num_classes))
    # logger.info(test(test_dataloader, model, loss_fn, opt, num_classes))
    visualize(model, x, y, num_classes, 'result_mDice.png', opt)
    # logger.info('Model with best mIou:')
    model.load_state_dict(torch.load(opt.save_path_iou))
    # logger.info(test(val_dataloader, model, loss_fn, opt, num_classes))
    # logger.info(test(test_dataloader, model, loss_fn, opt, num_classes))
    visualize(model, x, y, num_classes, 'result_mIou.png', opt)
    


if __name__ == '__main__':
    opt = parse_opt()
    # 目录
    make_directory(opt.running_dir)
    # 记录
    logger = init_file_logger(file=f'{opt.running_dir}/result.txt', name='my-logger')
    main(opt)
