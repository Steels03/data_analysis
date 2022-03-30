import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession, functions
from pyspark.sql.functions import regexp_extract
import matplotlib.pyplot as plt
from os.path import exists

spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()

data_file = "td/sample_access_log"

if (not exists(data_file)):
    print("ERROR : " + data_file + " isn't present")
    exit(1)
df = spark.read.text(data_file)
df.printSchema()
print((df.count(), len(df.columns)))
df.show(10, truncate=False)
#sample_logs = [item['value'] for item in df.take(15)]
#print(sample_logs)
ts_pattern = r'\[\d{2}\/\w{3}\/\d{4}:(\d{2}):\d{2}:\d{2} \+\d{4}\]'
timestamps_df = df.select(regexp_extract('value', ts_pattern, 1).cast("int").alias('timestamp'))
timestamp_freq_df = (timestamps_df.groupBy('timestamp').count().sort('timestamp').cache())
timestamp_freq_pd_df = (timestamp_freq_df.toPandas().sort_values(by=['timestamp'],ascending=True))
#print (timestamp_freq_pd_df)
#sns.catplot(y='count', data=timestamp_freq_pd_df, kind='bar', order=timestamp_freq_pd_df['timestamp'])
plt.step(
    x = timestamp_freq_pd_df['timestamp'],
    y = timestamp_freq_pd_df['count']
)
plt.show()
plt.figure.savefig('output.png')