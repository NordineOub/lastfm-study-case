from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.window import Window


def create_spark_session(app_name: str = "lastfm_sessions") -> SparkSession:
    """Create and configure a local SparkSession suitable for notebook use."""
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        # Fewer shuffle partitions is usually better on a laptop
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    return spark


def load_track_data(spark: SparkSession, path: str) -> DataFrame:
    """Load the lastfm track data from TSV with a fixed schema and parsed timestamp."""
    schema_def = StructType(
        [
            StructField("userid", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("musicbrainz_artist_id", StringType(), True),
            StructField("artist_name", StringType(), True),
            StructField("musicbrainz_track_id", StringType(), True),
            StructField("track_name", StringType(), True),
        ]
    )

    df = (
        spark.read.option("sep", "\t")
        .option("header", "false")
        .schema(schema_def)
        .csv(path)
    )

    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    return df


def add_sessions_id_columns(
    df: DataFrame,
    session_gap_sec: int,
    user_col: str = "userid",
    ts_col: str = "timestamp",
) -> DataFrame:
    """Add a session_id column based on gaps between consecutive plays per user."""
    w = Window.partitionBy(user_col).orderBy(ts_col)

    df = df.withColumn("prev_ts", F.lag(ts_col).over(w)).withColumn(
        "gap_flag",
        F.when(
            (F.col("prev_ts").isNotNull())
            & (
                (F.col(ts_col).cast("long") - F.col("prev_ts").cast("long"))
                > session_gap_sec
            ),
            1,
        ).otherwise(0),
    )

    w2 = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    df = df.withColumn("session_id", F.sum("gap_flag").over(w2))
    return df.drop("prev_ts", "gap_flag")
