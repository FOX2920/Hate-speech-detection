SCALE_VERSION = "2.13"
SPARK_VERSION = "3.5.0"
KAFKA_VERSION = "3.6.1"


DELAY = 5 # seconds
NUM_PARTITIONS = 3
KAFKA_BROKER = "localhost:9092"
STORE_TOPIC = "hate_speech_detect"
STORE_CONSUMER_GROUP = "hate_speech_detect"
PREDICTION_TOPIC = "prediction_hate_speech"
PREDICTION_CONSUMER_GROUP = "prediction_hate_speech"
JSON_SCHEMA_LIST = [
    "free_text STRING",
    "label_id INT"
]
