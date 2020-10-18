# -*- coding: utf-8 -*-
import pandas as pd
from fbprophet import Prophet
from datetime import  date,datetime,timedelta
from pyspark.sql.functions import col,length,collect_list,struct
from pyspark.sql import SparkSession
import logging
import sys

def create_dataframe(lag_day,spark):
    '''
    :param lag_day: 前n天
    :return: spark.DataFrame
    '''
    now = date.today()
    history_path = [(now - timedelta(days=i)).strftime("%Y/%m/%d") for i in range(lag_day)]
    history_path=list(map(lambda x: "/user/iptvqoe/privatedata/multi_dimension/area_all/mi/"+x+"/*",history_path))
    cols = ['area_code', 'record_date', 'count_time_type', 'online_users', 'play_users', 'good_num', 'over_loss_num',
            'over_lag_num','swtime_num', 'good_swtime_num', 'live_swtime_num', 'good_live_swtime_num', 'areaLagTime', 'areaPlayTime','ts_num']
    df = spark.read.format("csv").option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").load(path=history_path,header=False, sep='|')
    df = df.toDF(*cols)
    df = df.select("area_code", "record_date", "play_users")
    return df

def Clean(df,lag_day,nan_rate=0.6):
    # 过滤id
    df=df.where((length(col("area_code"))==6 ))
    ## 过滤缺失过多数据
    nan_count_threshold=lag_day*288*nan_rate
    filter_area_code=df.groupBy("area_code").count().filter("count >={0}".format(nan_count_threshold)).select("area_code")
    df=df.join(filter_area_code,on=["area_code"],how='inner')
    #排序
    df=df.sort(["area_code", "record_date"])
    return df


def run_prophet(x):
    """
     运行prophet
    :param x: pandas.DataFrame
    :return: pandas.DataFrame
    """
    x.columns=["ds","y"]
    x['ds']=x['ds'].map(lambda x: datetime.strptime(str(x),"%Y%m%d%H%M"))
    m=Prophet(daily_seasonality=True,weekly_seasonality=False,yearly_seasonality=False)
    m.fit(x)
    now = date.today()
    past_day=(now).day
    past_day_data_count=len(x[x.ds.map(lambda x:x.day==past_day)])
    future= m.make_future_dataframe(periods=288*2-past_day_data_count,freq="5min",include_history=False).tail(288)
    forcast=m.predict(future)
    forcast=forcast[['ds','yhat_lower']]
    forcast['ds']=forcast['ds'].map(lambda x: x.strftime("%Y%m%d%H%M"))
    return  forcast.to_numpy().tolist()

def make_forecast(x):
    """
    产生预测值
    :param x: list
    :return:  list
    """
    area_code=x['area_code']
    ds_y=x['ds_y']
    ds_y=list(map(lambda x: [int(x['record_date']), int(x['play_users'])], ds_y))
    ds_y=pd.DataFrame(ds_y)
    forecast=run_prophet(ds_y)
    return {"area_code": area_code, "forecast": forecast}

def agg(df):
    '''
    整合
    :param df:
    :return:
    '''
    group=df.groupBy("area_code")
    group=group.agg(collect_list(struct("record_date", "play_users").alias('data')))
    cols=['area_code','ds_y']
    group =  group.toDF(*cols)
    result=group.rdd.map(lambda x:make_forecast(x))
    return result.toDF()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    output_uri = "mongodb://testUser:test#mongo2O1g@10.135.32.22:27017/test.ResultProphet"
    input_uri = "mongodb://testUser:test#mongo2O1g@10.135.32.22:27017/test.ResultProphet"
    spark = SparkSession.builder.appName("prophet5min").config("spark.mongodb.output.uri", output_uri).config("spark.mongodb.input.uri", input_uri).enableHiveSupport().getOrCreate()
    lag_day=3
    logger.debug('=======载入数据=======')
    df=create_dataframe(lag_day,spark)
    logger.debug('clean')
    df=Clean(df, lag_day, nan_rate=0.65)
    logger.debug('predict')
    result=agg(df)
    result.write.format('com.mongodb.spark.sql.DefaultSource').mode('overwrite').save()
    now = date.today()+timedelta(days=1)
    history_path =now.strftime("%Y/%m/%d")
    result.write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").mode("overwrite").options(header="true").json("/user/iptvqoe/privatedata/alarm2/testEnv/AreaUserResult/"+ history_path)


if __name__ == '__main__':
    main()

'''
spark-submit \
--master yarn-cluster \
--queue queue_iptvqoe  \
--num-executors 10 \
--driver-memory 4g \
--executor-cores 6 \
--executor-memory 9g \
--conf spark.driver.port=50001 \
--conf spark.yarn.dist.archives=hdfs:///user/iptvqoe/spark/envs.zip#py_envs \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=py_envs/envs/py3.6/bin/python \
 /slview/qoezy/lushun1/area_al/prophet_area.py
'''