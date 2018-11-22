#!/bin/bash
# launch jupyter notebook with BDL

source hadoop-setconf.sh hadalytic

export SPARK_HOME=/afs/cern.ch/work/m/migliori/public/spark-2.3.1
export BIGDL_HOME=/afs/cern.ch/work/m/migliori/public/BDLtest/bigDL
export PYSPARK_PYTHON=/afs/cern.ch/work/m/migliori/public/anaconda2/bin/python

${BIGDL_HOME}/bin/jupyter-with-bigdl.sh \
--master local[32] \
--conf spark.driver.memory=250G