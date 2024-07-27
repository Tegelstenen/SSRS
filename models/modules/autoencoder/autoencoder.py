from keras import Model
from keras.layers import Input, Dense, Reshape
from keras.optimizers.legacy import Adam
from keras.losses import MeanAbsoluteError
import numpy as np
import os
import pickle

class Autoencoder:

    def __init__(self,
                 input_shape,
                 hidden_layers,
                 latent_space_dim):
        self.input_shape = input_shape # (rows, parameters)
        self.hidden_layers = hidden_layers # [128, 64]
        self.latent_space_dim = latent_space_dim # 2

        self.encoder = None
        self.decoder = None
        self.model = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanAbsoluteError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, x_test, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       validation_data=(x_test, x_test),
                       shuffle=True)

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, data):
        latent_representations = self.encoder.predict(data)
        reconstructed_data = self.decoder.predict(latent_representations)
        return reconstructed_data, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.hidden_layers,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layers = self._add_dense_layers(decoder_input, self.hidden_layers[::-1])
        decoder_output = self._add_decoder_output(dense_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layers(self, x, layers):
        for i, units in enumerate(layers):
            x = Dense(units, activation='relu', name=f"decoder_dense_{i+1}")(x)
        return x

    def _add_decoder_output(self, x):
        output_layer = Dense(np.prod(self.input_shape), activation="sigmoid", name="decoder_output")(x)
        return Reshape(self.input_shape)(output_layer)

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        dense_layers = self._add_dense_layers(encoder_input, self.hidden_layers)
        bottleneck = self._add_bottleneck(dense_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_dense_layers(self, x, layers):
        for i, units in enumerate(layers):
            x = Dense(units, activation='relu', name=f"encoder_dense_{i+1}")(x)
        return x

    def _add_bottleneck(self, x):
        bottleneck = Dense(self.latent_space_dim, name="encoder_output")(x)
        return bottleneck


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28),  # Example input shape
        hidden_layers=[128, 64],
        latent_space_dim=2
    )
    autoencoder.summary()