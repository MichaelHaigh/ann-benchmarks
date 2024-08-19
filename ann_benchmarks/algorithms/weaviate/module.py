import subprocess
import sys
import uuid

import weaviate
import weaviate.classes.config as wc

from ..base.module import BaseANN


class Weaviate(BaseANN):
    def __init__(self, metric, max_connections, ef_construction=512):
        self.class_name = f"Vector_{metric}_{max_connections}_{ef_construction}"
        self.client = weaviate.connect_to_custom(
            http_host="10.20.0.8",
            http_port="80",
            http_secure=False,
            grpc_host="10.20.0.9",
            grpc_port="50051",
            grpc_secure=False,
        )
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.distance = {
            "angular": wc.VectorDistances.COSINE,
            "euclidean": wc.VectorDistances.L2_SQUARED,
        }[metric]

    def fit(self, X):
        collection = self.client.collections.create(
            name=self.class_name,
            properties=[wc.Property(name="i", data_type=wc.DataType.INT)],
            vector_index_config=wc.Configure.VectorIndex.hnsw(
                distance_metric=self.distance,
                ef_construction=self.ef_construction,
                max_connections=self.max_connections,
            ),
        )
        with collection.batch.dynamic() as batch:
            batch.batch_size = 100
            for i, x in enumerate(X):
                properties = {"i": i}
                batch.add_object(properties=properties, uuid=uuid.UUID(int=i), vector=x)

    def set_query_arguments(self, ef):
        self.ef = ef
        collection = self.client.collections.get(self.class_name)
        collection.config.update(vectorizer_config=wc.Reconfigure.VectorIndex.hnsw(ef=ef))

    def query(self, v, n):
        collection = self.client.collections.get(self.class_name)
        ret = collection.query.near_vector(
            near_vector=v,
            limit=n,
        )
        # QueryReturn(objects=[Object(uuid=_WeaviateUUIDInt('00000000-0000-0000-0000-0000000ce915') ...])

        return [res.uuid.int for res in ret.objects]

    def __str__(self):
        return f"Weaviate(ef={self.ef}, maxConnections={self.max_connections}, efConstruction={self.ef_construction})"
