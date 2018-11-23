from bigdl.util.common import *
from bigdl.nn.layer import Model
from bigdl.dlframes.dl_classifier import DLModel
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import argparse
import sys

sns.set(style="darkgrid")

def loadParquet(spark, filePath, feature, label):
    df = spark.read.format('parquet') \
            .load(filePath) \
            .select([feature, label]) \
            .withColumnRenamed(feature, 'features')
    return df

def loadModel(modelPath):
    model = Model.loadModel(modelPath=modelPath+'.bigdl', weightPath=modelPath+'.bin')
    return model

def computeAUC(y_true, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr,tpr,roc_auc 

def savePlot(plotDir, values, modelType):
    plt.figure()
    for fpr,tpr,roc_auc in values:
        plt.plot(fpr[0], tpr[0],
            lw=2, label=modelType+' classifier (AUC) = %0.4f' % roc_auc[0])
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background Contamination (FPR)')
    plt.ylabel('Signal Efficiency (TPR)')
    plt.title('$tt$ selector')
    plt.legend(loc="lower right")
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.savefig(plotDir+'/auc.pdf')


def inference(spark, args):
    
    ## Lists containing fpr, tpr, auc for each model
    results = []

    for model in args.models:
        ## Check the type of model
        featureCol = ''
        label = 'encoded_label'
        featureSize = None 
        modelType = ''

        if 'hlf' in model:
            modelType = 'HLF'
            featureCol = 'HLF_input'
            featureSize = [14] 
        elif 'gru' in model:
            modelType = 'GRU'
            featureCol = 'GRU_input'
            featureSize = [801,19]
        else:
            sys.exit("Error, Invalid model type")

        testDF = loadParquet(spark, model, featureCol, label)
        model = loadModel(model)
        predictor = DLModel(model=model, featureSize=featureSize)
        predDF = predictor.transform(testDF)

        ## Collect results
        y_true = np.asarray(predDF.rdd \
                    .map(lambda row: row.__getitem__(featureCol)) \
                    .collect())
        y_pred = np.asarray(predDF.rdd \
                    .map(lambda row: row.prediction) \
                    .collect())

        results.append(computeAUC(y_true, y_pred))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--testDataset', type=str,
                        help ='path to the test dataset')
    parser.add_argument('--models', nargs='+', type=str,
                        help='List of models to compare')
    parser.add_argument('--plotDir', type=str,
                        help='Directory where to store plots')
    
    args = parser.parse_args()

    ## Create SparkContext and SparkSession
    sc = SparkContext(
        appName="Prediction",
        conf=create_spark_conf())
    spark = SQLContext(sc).sparkSession
    init_engine()
    sc.setLogLevel("ERROR")

    inference(spark,args)