#!/bin/bash
# Load a trained model and do inference

source hadoop-setconf.sh hadalytic

export SPARK_HOME=/afs/cern.ch/work/m/migliori/public/spark-2.3.1
export BIGDL_HOME=/afs/cern.ch/work/m/migliori/public/BDLtest/bigDL
export PYSPARK_PYTHON=/afs/cern.ch/work/m/migliori/public/anaconda2/bin/python

${BIGDL_HOME}/bin/spark-submit-with-bigdl.sh \
--master local[32] \
--conf spark.driver.memory=125G \
inference.py \
--testDataset hdfs://hadalytic/project/ML/data/test20k.parquet \
--models hlfHadalytic_hlf_10exe_6cores gruHadalytic_gru_11exe_6cores \
--modelsDir file:///afs/cern.ch/work/m/migliori/public/BDLtest/pythonScripts/models \
--plotDir /afs/cern.ch/work/m/migliori/public/BDLtest/pythonScripts/plots 