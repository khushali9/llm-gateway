from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, Int64, String

user = Entity(
    name        = "user_id",
    description = "LLM gateway user identifier",
)

request_features_source = FileSource(
    path            = "/home/ubuntu/llm-gateway/data/delta/request_features",
    timestamp_field = "feature_timestamp",
)

request_features_view = FeatureView(
    name     = "request_features",
    entities = [user],
    ttl      = timedelta(hours=24),
    schema   = [
        Field(name="prompt_length",         dtype=Int32),
        Field(name="prompt_token_estimate",  dtype=Int32),
        Field(name="max_tokens",            dtype=Int32),
        Field(name="domain_category",       dtype=String),
        Field(name="has_latency_slo",       dtype=Int32),
        Field(name="is_premium",            dtype=Int32),
        Field(name="latency_slo_ms",        dtype=Int32),
        Field(name="task_type",             dtype=String),
        Field(name="user_tier",             dtype=String),
        Field(name="kafka_partition",       dtype=Int32),
        Field(name="kafka_offset",          dtype=Int64),
    ],
    source   = request_features_source,
    online   = True,
)
