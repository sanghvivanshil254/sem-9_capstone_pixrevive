from abc import ABC, abstractmethod
import numpy as np
from ai.pipeline.pipeline_context import PipelineContext


class BaseModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        pass

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model weights into the specified device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free memory."""
        pass


class BaseRestorer(BaseModel):
    @abstractmethod
    def restore(self, context: PipelineContext) -> np.ndarray:
        """Perform restoration on the working_image in context."""
        pass


class BaseDetector(BaseModel):
    @abstractmethod
    def detect(self, context: PipelineContext) -> np.ndarray:
        """Perform detection (e.g., damage masks)."""
        pass


