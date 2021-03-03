"""
日志回滚的意思是：
假设日志文件是log2.txt，maxBytes设置的是1*1024，就是1M.当该文件到达1M大小时，会把该文件重命名为log2.txt.1,然后新建一个log2.txt文件用来保存接下来的日志信息，以此类推。而backupCount参数就限定了最多有多少个这样的"备份文件"。
如本模块例子所示，当达到10个备份的时候，再写日志，如果log2.txt装不下了，那么就不会往文件里写信息
"""
import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
# 定义一个RotatingFileHandler，最多备份3个日志文件，每个日志文件最大1K
rHandler = RotatingFileHandler("log2.txt", maxBytes=1 , backupCount=10)
rHandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
rHandler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(rHandler)
logger.addHandler(console)

logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")