from bigdl.util.common import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from bigdl.nn.criterion import CategoricalCrossEntropy
import numpy as np
import argparse
import time

def loadParquet(spark, filePath, columns):
    df = spark.read.format('parquet') \
            .load(filePath) \
            .select(columns) 
    return df

def createSample(df, featureCol, labelCol):
    rdd = df.select([featureCol, labelCol]) \
            .map(lambda row: Sample.from_ndarray(
                np.asarray(row.__getitem__(featureCol)),
                np.asarray(row.__getitem__(labelCol)) + 1
            ))
    return rdd

def buildGRUModel():
    recurrent = Recurrent()
    recurrent.add(
        GRU(
            input_size=19,
            hidden_size=50,
            activation='tanh',
            p=0.2
            )
    )

    model = Sequential()
    model.add(recurrent)
    model.add(Select(2,-1))
    model.add(Linear(50,3))
    model.add(SoftMax())
    return model

def buildOptimizer(model, trainRDD, batchSize,
                   numEpochs, appName, logDir):
    optimizer = Optimizer(
        model = model,
        training_rdd = trainRDD,
        criterion = CategoricalCrossEntropy(),
        optim_method = Adam(),
        end_trigger = MaxEpoch(numEpochs),
        batch_size = batchSize
    )

    trainSummary = TrainSummary(log_dir=logDir,app_name=appName)
    optimizer.set_train_summary(trainSummary)
    return optimizer

def train(spark, args):
    
    sc=spark.sparkContext
    featureCol = 'GRUinput'
    labelCol = 'encoded_label'
    numExecutors = int(sc._conf.get('spark.executor.instances'))
    exeCores = int(sc._conf.get('spark.executor.cores'))

    ## Load the parquet
    trainDF = loadParquet(spark, args.dataset, [featureCol, labelCol])
    ## Convert in into an RDD of Sample
    trainRDD = createSample(trainDF, featureCol, labelCol)

    model = buildGRUModel()

    batchSize = args.batchMultiplier * numExecutors * exeCores
    appName = args.appName + "_exe:{}_cores:{}".format(numExecutors, exeCores)
    optimizer = buildOptimizer(
        model = model,
        trainRDD = trainRDD,
        batchSize = batchSize,
        numEpochs = args.numEpochs,
        appName = appName,
        logDir = args.logDir
    )

    ## Start training
    start = time.time()
    optimizer.optimize()
    stop = time.time()

    print("\n\n Training time: {}\n\n".format(stop-start))

    if args.test == False: 
        with open('gruTimes.csv', 'a') as file:
            file.write("{},{},{},{}".format(
                exeCores,
                numExecutors,
                args.numEpochs,
                stop-start
                )
            )

    if args.saveModel == True:
        model.saveModel(
            modelPath = args.modelDir + '/' + appName + '.bigdl',
            weightPath = args.modelDir + '/' + appName + 'bin',
            over_write = True
        )
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchMultiplier', type=int, nargs=1)
    parser.add_argument('--numEpochs', type=int, nargs=1)
    parser.add_argument('--dataset', type=str, nargs=1)
    parser.add_argument('--jobName', type=str, nargs=1)
    parser.add_argument('--logDir', type=str, nargs=1)
    parser.add_argument('--saveModel', type=bool, nargs='?', const=False)
    parser.add_argument('--modelDir', type=str, nargs='?', const='~/')
    parser.add_argument('--test', type=bool, nargs='?', const=True)

    args = parser.parse_args()

    ## Create SparkContext and SparkSession
    sc = SparkContext(
        appName="HLFclassifier",
        conf=create_spark_conf())
    spark = SQLContext(sc).sparkSession

    ## Init bigDL and show logs
    sc.setLogLevel("ERROR")
    show_bigdl_info_logs()
    init_engine()

    train(sc, args)

    


    
