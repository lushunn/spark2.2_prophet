# -*- coding: utf-8 -*-
from datetime import  date,datetime,timedelta
from pyspark.sql import SparkSession
import logging
def load_pass_dataframe(spark,lag_day=1):
    '''
    :param lag_day: 前n天
    :return: spark.DataFrame
    '''
    now = date.today()
    x = (now - timedelta(days=lag_day)).strftime("%Y/%m/%d")
    history_path="/user/iptvqoe/privatedata/multi_dimension/area_all/mi/"+x+"/*"
    cols = ['area_code', 'record_date', 'count_time_type', 'online_users', 'play_users', 'good_num', 'over_loss_num',
            'over_lag_num','swtime_num', 'good_swtime_num', 'live_swtime_num', 'good_live_swtime_num', 'areaLagTime', 'areaPlayTime','ts_num']
    df = spark.read.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format("csv").load(path=history_path,header=False, sep='|')
    df = df.toDF(*cols)
    df = df.select("area_code", "record_date", "play_users")
    df=df.withColumn("record_date", df["record_date"].cast("bigint"))
    df = df.withColumn("play_users", df["play_users"].cast("bigint"))
    return df

def load_predict(spark,lag_day=1):
    now = date.today()
    history_path= (now - timedelta(days=lag_day)).strftime("%Y/%m/%d")
    df=spark.read.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").format('json').load("/user/iptvqoe/privatedata/alarm2/testEnv/AreaUserResult/"+ history_path)
    def helper(x):
        area_code=x["area_code"]
        forecast=x["forecast"]
        time=list(map(lambda x: int(x[0]),forecast))
        value=list(map(lambda x: int(float(x[1])),forecast))
        areas=[area_code]*len(time)
        return list(zip(areas,time,value))
    return df.rdd.flatMap(helper).toDF(["area_code", "record_date","forcaste_value"])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    spark = SparkSession.builder.appName("verifiction").enableHiveSupport().getOrCreate()
    lag_day = 1
    logger.debug('=======载入真实数据=======')
    df = load_pass_dataframe(spark,lag_day)
    logger.debug('=======载预测数据=======')
    predict=load_predict(spark)
    logger.debug('=======连接=======')
    result = df.join(predict, on=["area_code", "record_date"], how="inner")
    result=result.withColumn("diff", result["play_users"]-result["forcaste_value"])
    now = date.today()-timedelta(lag_day)
    history_path =now.strftime("%Y/%m/%d")
    logger.debug('=======导出=======')
    result=result.where("diff<0")
    result.write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").mode("overwrite").options(header="true").csv("/user/iptvqoe/privatedata/alarm2/testEnv/AreaUserResultVerify/"+ history_path)
