from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder.appName('SpamClassifier').getOrCreate()

df = spark.read.option("header", "false") \
    .option("delimiter", "\t") \
    .option("inferSchema", "true") \
    .csv("td2/spam.txt") \
    .withColumnRenamed("_c0", "classification") \
    .withColumnRenamed("_c1", "sms_content")

df.limit(10).show()

stages = []

regexTokenizer = RegexTokenizer(inputCol="sms_content", outputCol="tokens", pattern="\\W+")
stages += [regexTokenizer]

cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)#, vocabSize=3, minDF=2.0
stages += [cv]

indexer = StringIndexer(inputCol="classification", outputCol="label")
stages += [indexer]

vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
stages += [vecAssembler]

[print('\n', stage) for stage in stages]

pipeline = Pipeline(stages=stages)
data = pipeline.fit(df).transform(df)
train, test = data.randomSplit([0.7, 0.3], seed = 2018)
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)

predictions = model.transform(test)
predictions.limit(100).select("label", "prediction", "probability").show(truncate=False)


# évaluation du modèle
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print ("Précision : ", accuracy)

# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
# Run Cross-validation
cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
cvModel = cv.fit(train)
# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
evaluator.evaluate(cvPredictions)

# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
print ("Test Area Under ROC: ", evaluator.evaluate(cvPredictions))

