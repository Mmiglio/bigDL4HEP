#!/bin/bash
# Submit bigDL job

source hadoop-setconf.sh hadalytic

export SPARK_HOME=/afs/cern.ch/work/m/migliori/public/spark-2.3.1
export BIGDL_HOME=/afs/cern.ch/work/m/migliori/public/BDLtest/bigDL
export PYSPARK_PYTHON=/afs/cern.ch/work/m/migliori/public/anaconda2/bin/python

${BIGDL_HOME}/bin/spark-submit-with-bigdl.sh \
--master yarn \
--conf spark.driver.memory=125G \
--conf spark.driver.cores=4 \
--conf spark.executor.instances=10 \
--conf spark.executor.cores=4 \
--conf spark.executor.memory=14G \
trainClassifier.py \
--model inclusive \
--batchMultiplier 32 \
--numEpochs 70 \
--dataset hdfs://hadalytic/project/ML/data/train100k.parquet \
--validation hdfs://hadalytic/project/ML/data/test20k.parquet \
--jobName adam_lr2_decay2_NODropout_BatchNormal_batch32_70epochs \
--logDir bigdl_summaries \
--saveModel True \
--modelDir /afs/cern.ch/work/m/migliori/public/BDLtest/pythonScripts/models \
--saveTime True 