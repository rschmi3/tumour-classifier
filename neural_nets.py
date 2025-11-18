import numpy as np
from keras import Model, callbacks, layers, losses, models


class Conv2dMaxPoolingLayer(Model):
    def __init__(
        self,
        filters: int,
        kernel_size: tuple[int, int] = (3, 3),
        pool_size: tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
        )
        self.max = layers.MaxPooling2D(pool_size=pool_size)
        self.dropout = layers.Dropout(0.25)

    def build(self, input_shape):
        # Build the sub-layers
        self.conv.build(input_shape)
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.max.build(conv_output_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.max(x)
        if training:
            x = self.dropout(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        # Chain the output shape computations
        shape = self.conv.compute_output_shape(input_shape)
        shape = self.max.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
            }
        )
        return config


class TumourNet(Model):
    def __init__(self, use_augmentation: bool = True):
        super().__init__()
        # Data augmentation layers (only active during training)
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augmentation = [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),  # Rotate up to 15%
                layers.RandomZoom(0.1),  # Zoom up to 10%
                layers.RandomBrightness(0.2),  # Adjust brightness
                layers.RandomContrast(0.2),  # Adjust contrast
            ]

        self.rescaling = layers.Rescaling(1.0 / 255)
        self.conv1 = Conv2dMaxPoolingLayer(32, (3, 3))
        self.conv2 = Conv2dMaxPoolingLayer(64, (3, 3))
        self.conv3 = Conv2dMaxPoolingLayer(64, (3, 3))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(4, activation="softmax")

    def call(self, x, training=None):
        # Apply augmentation only during training
        if self.use_augmentation and training:
            for aug_layer in self.augmentation:
                x = aug_layer(x, training=True)

        x = self.rescaling(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class TumourNetWrapper:
    """Wrapper for keras model to add functions that mimic scikit learn model class structure"""

    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs):
        self.model.fit(
            X,
            y,
            epochs=epochs,
            callbacks=[
                callbacks.LearningRateScheduler(lambda epoch: 0.001 * (0.98**epoch))
            ],
        )

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

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
