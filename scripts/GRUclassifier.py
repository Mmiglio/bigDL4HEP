from bigdl.util.common import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from bigdl.nn.criterion import CategoricalCrossEntropy
import numpy as np
import argparse

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

    return optimizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs=1)

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

    


    
