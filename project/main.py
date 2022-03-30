import constants
from os.path import exists
import findspark
import pyspark
from os.path import exists
from functools import reduce
from pyspark.ml.feature import CountVectorizer, RegexTokenizer, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from project.code_remi import load_dataset

findspark.init()

check_dataset_exists(constants.TRAIN_DATASET_PATH)
check_dataset_exists(constants.TRAIN_20_PERCENT_DATASET_PATH)
check_dataset_exists(constants.TEST_DATASET_PATH)

spark = pyspark.sql.SparkSession.builder.appName("project").getOrCreate()

def check_dataset_exists(path):
    """ check if dataset is present """
    if (not exists(path)):
        print("ERROR : " + path + " isn't present")
        exit(1)
    else:
        print("dataset is present")

df = spark.read.format("csv").load(constants.TRAIN_DATASET_PATH)
df.limit(10).show()
df.printSchema()

oldColumns = df.schema.names 

# On ajoute le nom des colonnes au dataframe
df = reduce(lambda df, idx: df.withColumnRenamed(oldColumns[idx], constants.COLUMN_NAMES[idx]),\
    range(len(oldColumns)), df)
df.printSchema()
df.limit(10).show()

df_train = load_dataset(constants.TEST_DATASET_PATH)
df_train_20_percent = load_dataset(constants.TRAIN_20_PERCENT_DATASET_PATH)
df_test = load_dataset(constants.TRAIN_DATASET_PATH)

stages = []
    # 1. clean data and tokenize sentences using RegexTokenizer

lst_output_colum_name = []
lst_tokens_feature_output_colum_name = []
for string_colum in ['protocol_type', 'service', 'flag', 'difficulty_group']:
    regexTokenizer = RegexTokenizer(inputCol=string_colum, outputCol=string_colum+"_tokens", pattern="\\W+")
    stages += [regexTokenizer]
    lst_output_colum_name.append(string_colum+"_tokens")


# 2. CountVectorize the data
for colum in lst_output_colum_name:
    cv = CountVectorizer(inputCol=colum, outputCol=colum+"_token_features", minDF=2.0)  # , vocabSize=3, minDF=2.0
    stages += [cv]
    lst_tokens_feature_output_colum_name.append(colum+"_token_features")

print("FIN de la vectorisation")
# 3. Convert the labels to numerical values using binariser
indexer = StringIndexer(inputCol="labels", outputCol="label")
stages += [indexer]

# 4. Vectorise features using vectorassembler
vecAssembler = VectorAssembler(inputCols=lst_tokens_feature_output_colum_name, outputCol="features")
stages += [vecAssembler]

[print('\n', stage) for stage in stages]

pipeline = Pipeline(stages=stages)
data = pipeline.fit(df_train).transform(df_train)

train, test = data.randomSplit([0.7, 0.3], seed=2018)

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)

predictions = model.transform(test)
# Select results to view
predictions.limit(10).select("label", "prediction", "probability").show(truncate=False)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: ", accuracy)

string_indexers = [is_attack_indexer, type_attack_indexer]
one_hot_encoders = []
for col in nominal_cols:
    col_idx = "{0}_idx".format(col)
    col_vec = "{0}_vec".format(col)
    string_indexers.append(StringIndexer(inputCol=col, outputCol=col_idx))
    one_hot_encoders.append(OneHotEncoder(inputCol=col_idx, outputCol=col_vec))
    if col_vec not in binary_cols:
        binary_cols.append(col_vec)
numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol='num_feats')
numeric_scaler = StandardScaler(inputCol="num_feats", outputCol="scaled_num_feats", withStd=True, withMean=False)

global_assembler = VectorAssembler(inputCols=binary_cols + ["scaled_num_feats"], outputCol='raw_feats')
global_indexer = VectorIndexer(inputCol='raw_feats', outputCol='idx_feats', maxCategories=2)

preparation_pipeline = Pipeline(stages=string_indexers + one_hot_encoders + [numeric_assembler, numeric_scaler, global_assembler, global_indexer])
prep_model = preparation_pipeline.fit(train_df)

prep_train_df = prep_model.transform(train_df) \
    .select('idx_feats', 'is_attack_idx', 'is_attack', 'type_attack_idx', 'type_attack') \
    .cache()
prep_test_df = prep_model.transform(test_df) \
    .select('idx_feats', 'is_attack_idx', 'is_attack', 'type_attack_idx', 'type_attack') \
    .cache()

split = (prep_train_df.randomSplit([0.8, 0.2]))

final_train_df = split[0].cache()
final_cv_df = split[1].cache()
final_test_df = prep_test_df

rf_t0 = time()
rf_is_attack_classifier = RandomForestClassifier(
    labelCol='is_attack_idx', featuresCol='idx_feats', predictionCol="is_attack_pred_idx",
    numTrees=100,
    maxDepth=10,
    featureSubsetStrategy="sqrt")

rf_is_attack_pipeline = Pipeline(stages=[rf_is_attack_classifier, is_attack_converter])
rf_is_attack_model = rf_is_attack_pipeline.fit(final_train_df)
print("{0} secs".format(time() - rf_t0))

cv_is_attack_predictions = rf_is_attack_model.transform(final_cv_df)
test_is_attack_predictions = rf_is_attack_model.transform(final_test_df)

cv_is_attack_predictions.show()

is_attack_cv_accuracy = MulticlassClassificationEvaluator(labelCol="is_attack_idx", predictionCol="is_attack_pred_idx", metricName="accuracy").evaluate(cv_is_attack_predictions)
print("is_attack Cross-validation Error = %g" % (1.0 - is_attack_cv_accuracy))

is_attack_test_accuracy = MulticlassClassificationEvaluator(labelCol="is_attack_idx", predictionCol="is_attack_pred_idx", metricName="accuracy").evaluate(test_is_attack_predictions)
print("Ratio d'erreur sur les prédictions d'attaques = %g" % (1.0 - is_attack_test_accuracy))
