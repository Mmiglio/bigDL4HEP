from bigdl.util.common import *
from bigdl.optim.optimizer import *
from bigdl.nn.criterion import CategoricalCrossEntropy
import models
import numpy as np
import argparse
import time

def loadParquet(spark, filePath, featureCol, labelCol, sample, frac):
    ## If frac!=0 sample the dataframe, otherwise return the full df
    if sample:
        df = spark.read.format('parquet') \
            .load(filePath) \
            .select(featureCol + [labelCol]) \
            .sample(False, frac, 42)
    else:
        df = spark.read.format('parquet') \
            .load(filePath) \
            .select(featureCol + [labelCol])    
    return df

def createSample(df, featureCol, labelCol):
    rdd = df.rdd \
            .map(lambda row: Sample.from_ndarray(
                [np.asarray(row.__getitem__(feature)) for feature in featureCol],
                np.asarray(row.__getitem__(labelCol)) + 1
            ))
    return rdd

def buildOptimizer(model, trainRDD, valRDD, batchSize,
                   numEpochs, appName, logDir):
    optimizer = Optimizer(
        model = model,
        training_rdd = trainRDD,
        criterion = CategoricalCrossEntropy(),
        optim_method = Adam(learningrate=0.002, learningrate_decay=0.0002, epsilon=9e-8),
        end_trigger = MaxEpoch(numEpochs),
        batch_size = batchSize   
    )

    ## Remove the already existing job logs
    import os
    try:
        os.system('rm -rf '+logDir+'/'+appName)
    except:
        pass

    if valRDD != False:
        optimizer.set_validation(
            batch_size = batchSize,
            val_rdd = valRDD,
            trigger = EveryEpoch(),
            val_method=[Loss(CategoricalCrossEntropy())]
        )
        valSummary = ValidationSummary(log_dir=logDir,app_name=appName)
        optimizer.set_val_summary(valSummary)
        
    trainSummary = TrainSummary(log_dir=logDir,app_name=appName)
    optimizer.set_train_summary(trainSummary)
    
    return optimizer

def train(spark, args):
    
    sc=spark.sparkContext
    numExecutors = int(sc._conf.get('spark.executor.instances'))
    exeCores = int(sc._conf.get('spark.executor.cores'))
    
    labelCol = 'encoded_label'
    if args.model == 'gru':
        featureCol = ['GRU_input']
        model = models.GRUModel()
    elif args.model == 'hlf':
        featureCol = ['HLF_input']
        model = models.HLFmodel()
    elif args.model == 'inclusive':
        featureCol = ['GRU_input', 'HLF_input']
        model = models.InclusiveModel()
    else:   
        sys.exit("Error, insert a valid model!")
   
    ## Load the parquet
    if args.frac != 1:
        sampleDF = True
    else:
        sampleDF = False

    trainDF = loadParquet(spark, args.dataset, featureCol, labelCol, sample=sampleDF, frac=args.frac)
    ## Convert in into an RDD of Sample
    trainRDD = createSample(trainDF, featureCol, labelCol)

    if args.validation != 'False':
        testDF = loadParquet(spark, args.validation, featureCol, labelCol, sample=False, frac=args.frac)
        testRDD = createSample(testDF, featureCol, labelCol)
    else:
        testRDD = False

    batchSize = args.batchMultiplier * numExecutors * exeCores
    appName = args.jobName + "_" + args.model + "_{}exe_{}cores".format(numExecutors, exeCores)
    

    optimizer = buildOptimizer(
        model = model,
        trainRDD = trainRDD,
        valRDD = testRDD,
        batchSize = batchSize,
        numEpochs = args.numEpochs,
        appName = appName,
        logDir = args.logDir
    )

    ## Start training
    start = time.time()
    optimizer.optimize()
    stop = time.time()

    print("\n\n Elapsed time: {:.2f}s\n\n".format(stop-start))

    if args.saveModel == True:
        model.saveModel(
            modelPath = args.modelDir + '/' + appName + '.bigdl',
            weightPath = args.modelDir + '/' + appName + '.bin',
            over_write = True
        )

    if args.saveTime == True: 
        with open(args.model+'Times.csv', 'a') as file:
            file.write("{},{},{},{},{},{:.2f}\n".format(
                args.batchMultiplier,
                batchSize,
                exeCores,
                numExecutors,
                args.numEpochs,
                stop-start
                )
            )
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, nargs='?', const='gru',
                        help="Choose a model between gru/..")
    parser.add_argument('--batchMultiplier', type=int, nargs='?', const=16)
    parser.add_argument('--numEpochs', type=int, nargs='?', const=50)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--frac', type=float, nargs='?', const=1,
                        help='fraction of the dataset to be used')
    parser.add_argument('--validation', type=str, nargs='?', const='False',
                        help='False=No validation, otherwise give the path to the dataset')
    parser.add_argument('--jobName', type=str)
    parser.add_argument('--logDir', type=str,
                        help="Directory containing logs,losses and throughputs")
    parser.add_argument('--saveModel', type=bool, nargs='?', const=False,
                        help="If True the model will be saved in modelDir")
    parser.add_argument('--modelDir', type=str, nargs='?', const='~/',
                        help="Path to the directory where to store the model")
    parser.add_argument('--saveTime', type=bool, nargs='?', const=False,
                        help="Save the elapsed time in a csv file")

    args = parser.parse_args()
    
    ## Create SparkContext and SparkSession
    sc = SparkContext(
        appName="BDLclassifier",
        conf=create_spark_conf())
    spark = SQLContext(sc).sparkSession

    ## Init bigDL and show trainign logs on the terminal
    sc.setLogLevel("ERROR")
    show_bigdl_info_logs()
    init_engine()

    train(spark, args)

    print('Done! So long and thanks for all the fish.')

    


    
