import logging


def init_file_logger(file='./logs/log.txt', name='my-logger', level=logging.DEBUG) -> logging.Logger:
    """
    初始化文件日志记录器
    :param path: 日志文件存放目录
    :param prefix: 日志文件名前缀
    :param name: 日志记录器名称
    :param level: 日志级别
    :return: 日志记录器（单例、全局唯一）
    """
    # 日志记录器
    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    # 文件处理器
    file_handler = logging.FileHandler(file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_handler_formatter)
    logger.addHandler(file_handler)
    # 文件处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_handler_formatter)
    logger.addHandler(stream_handler)
    return logger


if __name__ == '__main__':
    logger = init_file_logger(name='test')
    logger.info('test1')
    logger.warning('test2')
    logger.error('test3')