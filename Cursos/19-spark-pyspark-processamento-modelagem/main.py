# %%
import sys
!{sys.executable} -m pip install pyspark

# %%
from pyspark.sql import SparkSession

# %%
spark = SparkSession.builder \
    .appName("PrimeirosPassos") \
    .getOrCreate()

# Agora seu código vai funcionar:
df = spark.createDataFrame([("hello", "world")], ["col1", "col2"])
df.show()

# %%
# CURSO CHATO!