import logging
import torch


def logger_config(log_path, logging_name=None):
    """
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    """
    # logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）

    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 移除所有现有的handlers，确保不会重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


# 定义保存checkpoint的函数
def save_checkpoint_model(fold, model, optimizer,loss, scheduler, path):
    torch.save({
        'fold': fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,  # 可以添加其他需要保存的信息，如当前的loss值
    }, path)


# 定义加载checkpoint的函数
def load_checkpoint_model(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    fold = checkpoint['fold']
    return model, optimizer, fold, loss