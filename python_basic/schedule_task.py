#定时任务

#定义任务执行的次数
count = 0
#定义一个任务
def task():
    print("第%d次执行任务"%count)

#定时执行任务

#方法1
import time
def method1():
    while True:
        global count
        count += 1
        task()
        time.sleep(2)

#方法2
#APScheduler是一个 Python 定时任务框架，使用起来十分方便。提供了基于日期、固定时间间隔以及 crontab类型的任务，并且可以持久化任务、并以 daemon 方式运行应用。
from apscheduler.schedulers.blocking import BlockingScheduler
def blocking_scheduler():
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    # 参数：要执行的函数，cron是持续百万年的意思，day_of_week是在一周的哪几天，hour和minute是几点几分
    scheduler.add_job(task, 'cron', day_of_week='1-5', hour=15, minute=54)
    scheduler.start()

# from apscheduler.schedulers.tornado import TornadoScheduler
# def tornado_scheduler():
#     scheduler = TornadoScheduler(timezone="Asia/Shanghai")
#     scheduler.add_job(task, 'cron', day_of_week='1-5',hour=17)
#     scheduler.start()
# tornado_scheduler()
