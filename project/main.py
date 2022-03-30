import constants
from os.path import exists
import findspark
import pyspark
from os.path import exists
import numpy as np
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from functools import reduce

findspark.init()

dataset_file = "project/dataset/KDDTrain+.txt"

spark = pyspark.sql.SparkSession.builder.appName("project").getOrCreate()

#check if dataset is present
if (not exists(dataset_file)):
    print("ERROR : " + dataset_file + " isn't present")
    exit(1)
else:
    print("dataset is present")

df = spark.read.format("csv").load(dataset_file)
df.limit(10).show()
df.printSchema()

oldColumns = df.schema.names 

df = reduce(lambda df, idx: df.withColumnRenamed(oldColumns[idx], constants.COLUMN_NAMES[idx]), range(len(oldColumns)), df)
df.printSchema()
df.limit(10).show()

stages = []
# 1. clean data and tokenize sentences using RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="sms", outputCol="tokens", pattern="\\W+")
stages += [regexTokenizer]

# 2. CountVectorize the data
cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)#, vocabSize=3, minDF=2.0
stages += [cv]

# 3. Convert the labels to numerical values using binariser
indexer = StringIndexer(inputCol="label_string", outputCol="label")
stages += [indexer]

# 4. Vectorise features using vectorassembler
vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
stages += [vecAssembler]

[print('\n', stage) for stage in stages]