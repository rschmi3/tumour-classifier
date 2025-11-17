import numpy as np
from keras import Model, layers, losses, models, saving


class Conv2dMaxPoolingLayer(Model):
    def __init__(
        self,
        filters: int,
        kernel_size: tuple[int, int] = (3, 3),
        pool_size: tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__()
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
        )
        self.max = layers.MaxPooling2D(pool_size=pool_size)

    def call(self, x):
        x = self.conv(x)
        x = self.max(x)
        return x


class TumourNet(Model):
    def __init__(self):
        super().__init__()

        self.rescaling = layers.Rescaling(1.0 / 255)
        self.conv1 = Conv2dMaxPoolingLayer(32, (3, 3))
        self.conv2 = Conv2dMaxPoolingLayer(64, (3, 3))
        self.conv3 = Conv2dMaxPoolingLayer(64, (3, 3))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(4, activation="softmax")

    def call(self, x):
        x = self.rescaling(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class TumourNetWrapper:
    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        if probs.shape[1] == 1:  # Binary
            return (probs > 0.5).astype(int).ravel()
        else:  # Multi-class
            return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs):
        self.model.fit(X, y, epochs=epochs)

    def compile(self):
        self.model.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def save(self, filename):
        self.model.save(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        model = models.load_model(filename)
        if isinstance(model, Model):
            self.model: Model = model

    def load_weights(self, filename):
        self.model.load_weights(filename)
