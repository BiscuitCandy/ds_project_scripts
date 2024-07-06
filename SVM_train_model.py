from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, RegexTokenizer, CountVectorizer, Word2Vec, Word2VecModel
import pandas as pd
from pyspark.sql.functions import when
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes,NaiveBayesModel
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator

def naiveBayes(dataset):
    nb = NaiveBayes(smoothing=1.0)
    nbModel = nb.fit(dataset)
    #nbModel.write().overwrite().save('./Models/newExpModels/NBModelSongsWithMD11')
    nbModel.save('./HcqNB')
    return nbModel

def logisticRegression(dataset):
    lr = LogisticRegression(maxIter=50)
    lrModel = lr.fit(trainingData)
    #output_dir = './LRModel2'
    lrModel.save('./HcqLR')
    return lrModel

def SVM(dataset):
    svm = LinearSVC(maxIter=10, regParam=0.1)
    lsvcModel = svm.fit(dataset)
    lsvcModel.save('./HcqSVM')
    return lsvcModel

def evaluation(predictions):
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR",rawPredictionCol="rawPrediction")
    accuracy = evaluator.evaluate(predictions)
    print("Model Accuracy: ", accuracy)
    #predictionRDD = predictions.select(['label', 'prediction'])
    #                        .rdd.map(lambda line: (line[1], line[0]))
    #metrics = MulticlassMetrics(predictionRDD)
    #print "%s" % metrics.precision()
    #print "%s" % metrics.weightedRecall

# initiate our session and read the main CSV file, then we print the #dataframe schema
spark = SparkSession.builder.appName('imbalanced_binary_classification').master('local[*]').config('job.local.dir','/home/vivek/testSpark/').getOrCreate()
#word2VecModel = Word2VecModel.load('word2vecAttrh_300.model')
#print(word2VecModel.getVectors())
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

pos = spark.read.csv("./dash_renal/positive_query1_data.csv", header=True)
neg = spark.read.csv("./dash_renal/negative_query1_data.csv", header=True)

df = pos.union(neg)

# print(data.head())
# data.rename(columns={'caption':'attr','meta_h':'name'}, inplace=True)
# data[['attr','name','label']] = data[['attr','name','label']].astype(str)
# print(data.head())

# data.dropna()
# #data.printSchema()
# print(data)
# df = spark.createDataFrame(data)
df = df.selectExpr("caption as attr","meta_h as name","cast(label as int) label")
df.printSchema()

df = df.na.drop()
#word2VecModel = Word2VecModel.load('word2vecAttrh_300.model')
#print(word2VecModel.getVectors())
#print df.groupby('label').count()

df = df.withColumn("label", when(df.label == 0, 0).otherwise(1))

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="name", outputCol="words")
regexTokenizer1 = RegexTokenizer(inputCol="attr", outputCol="attrWords")
#regexTokenizer.transform(df.na.drop(Array("words")))

# bag of words count
countVectors = CountVectorizer(inputCol="words", outputCol="wordfeatures", vocabSize=500000, minDF=5)
countVectors1 = CountVectorizer(inputCol="attrWords", outputCol="attrFeatures", vocabSize=500000, minDF=5)

#(trainingData, testData) = df.randomSplit([0.7, 0.3], seed = 100)
#trainingData.cache()
#testData.cache()

#nb = NaiveBayes(smoothing=1.0, modelType="bernoulli")
pipeline = Pipeline(stages=[regexTokenizer1,regexTokenizer,countVectors1,countVectors])
#pipeline = Pipeline(stages=[regexTokenizer,countVectors])
model = pipeline.fit(df)
model.save('./hcqSvm')
dataset = model.transform(df)

assembler = VectorAssembler(inputCols=['attrFeatures','wordfeatures'],outputCol="features")
#assembler = VectorAssembler(inputCols=['wordfeatures'],outputCol="features")
dataset = assembler.transform(dataset)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
#print("=======================================")
#dataset.show()
#print("=======================================")

trainingData.cache()
testData.cache()

# crossValidation(pipeline,dataset)
print (trainingData.count())
print (testData.count())

trainModel = SVM(dataset)
#trainModel = naiveBayes(dataset)
#trainModel = logisticRegression(dataset)

predictions = trainModel.transform(testData)

predictions.filter(predictions['prediction'] == 1) \
    .select("features","label","prediction") \
    .orderBy("prediction", ascending=False) \
    .show(n = 10, truncate = 30)

# Make predictions on testData so we can measure the accuracy of our model on new data
#predictions = model.transform(testData)

# Display what results we can view
predictions.printSchema()

# Compute raw scores on the test set
# predictionAndLabels = testData.map(lambda lp: predictions.select("prediction"), testData.label)

# metrics = BinaryClassificationMetrics(predictions.select("prediction"))
# Area under precision-recall curve
# print("Precision = %s" % metrics.precision)

evaluation(predictions)