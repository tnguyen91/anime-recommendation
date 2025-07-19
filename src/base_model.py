from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, train_data, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, user_data, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass