#!/bin/bash
# Submit bigDL job

source hadoop-setconf.sh hadalytic

export SPARK_HOME=/afs/cern.ch/work/m/migliori/public/spark-2.4
export BIGDL_HOME=/afs/cern.ch/work/m/migliori/public/BDLtest/bigDL
export PYSPARK_PYTHON=/afs/cern.ch/work/m/migliori/public/anaconda2/bin/python

${BIGDL_HOME}/bin/spark-submit-with-bigdl.sh \
--master yarn \
--conf spark.driver.memory=6G \
--conf spark.driver.cores=4 \
--conf spark.executor.instances=22 \
--conf spark.executor.cores=6 \
--conf spark.executor.memory=14G \
--conf spark.security.credentials.hadalytic.enabled=false \
--conf spark.dynamicallocation.enabled=false \
--conf spark.yarn.access.hadoopFileSystems=hadalytic \
--conf "spark.metrics.conf.*.sink.graphite.class"="org.apache.spark.metrics.sink.GraphiteSink" \
--conf "spark.metrics.conf.*.sink.graphite.host"=HOST-IP \
--conf "spark.metrics.conf.*.sink.graphite.port"=2003 \
--conf "spark.metrics.conf.*.sink.graphite.period"=10 \
--conf "spark.metrics.conf.*.sink.graphite.unit"=seconds \
--conf "spark.metrics.conf.*.sink.graphite.prefix"="matteo" \
--conf "spark.metrics.conf.*.source.jvm.class"="org.apache.spark.metrics.source.JvmSource" \
--conf spark.app.status.metrics.enabled=true \
trainClassifier.py \
--model inclusive \
--batchMultiplier 32 \
--numEpochs 20 \
--dataset hdfs://hadalytic/project/ML/data/trainUndersampled_v2_rep.parquet \
--frac 0.25 \
--validation hdfs://hadalytic/project/ML/data/test20k.parquet \
--jobName validation_frac0.25_20epochs_Hadalytic \
--logDir bigdl_summaries \
--saveModel True \
--modelDir /afs/cern.ch/work/m/migliori/public/BDLtest/pythonScripts/models \
--saveTime True \
