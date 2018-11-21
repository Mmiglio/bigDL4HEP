from bigdl.util.common import *

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs=1)

    args = parser.parse_args()

    ## Create SparkContext and SparkSession
    sc = SparkContext(
        appName="HLFclassifier",
        conf=create_spark_conf())
    spark = SQLContext(sc)

    ## Init bigDL and show logs
    sc.setLogLevel("ERROR")
    show_bigdl_info_logs()
    init_engine()
    


    
