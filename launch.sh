#!/bin/bash
# Submit bigDL job

source hadoop-setconf.sh hadalytic

export SPARK_HOME=/afs/cern.ch/work/m/migliori/public/spark-2.3.1
export BIGDL_HOME=/afs/cern.ch/work/m/migliori/public/BDLtest/bigDL
export PYSPARK_PYTHON=/afs/cern.ch/work/m/migliori/public/anaconda2/bin/python

${BIGDL_HOME}/bin/spark-submit-with-bigdl.sh \
--master yarn \
--conf spark.driver.memory=5G \
--conf spark.driver.cores=4 \
--conf spark.executor.instances=2 \
--conf spark.executor.cores=4 \
--conf spark.executor.memory=14G \
HLFclassifier.py \
--dataset file:///afs/cern.ch/work/m/migliori/test20k.parquet