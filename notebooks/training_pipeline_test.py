# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import json
from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import (
    LendingClubTrainingPipeline,
)

# COMMAND ----------

conf = json.load(open("../conf/lendingclub_config.json"))
p = LendingClubTrainingPipeline(spark, conf)
p.run()

# COMMAND ----------
