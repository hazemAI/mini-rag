from .providers import QdrantDBProvider
from .VectorDBEnums import VectorDBEnums, DistanceMethodEnums
from controllers.BaseController import BaseController


class VectorDBProviderFactory:
    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider):
        if provider == VectorDBEnums.QDRANT.value:
            db_path = self.base_controller.get_database_path(
                db_name=self.config.VECTOR_DB_PATH
            )
            distance_method = getattr(
                self.config,
                "VECTOR_DB_DISTANCE_METHOD",
                DistanceMethodEnums.COSINE.value,
            )
            return QdrantDBProvider(
                db_path=db_path,
                distance_method=distance_method
            )

        return None
