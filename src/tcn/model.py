import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Dense,
    Dropout,
    Add,
    Activation,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam


def residual_block(
    x,
    filters,
    kernel_size,
    dilation_rate,
    dropout_rate=0.15,
):
    """
    TCN residual block.

    Causal convolution ensures that predictions
    do not use future information.
    """

    shortcut = x

    # First causal dilated convolution
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)

    x = Activation(
        "relu"
    )(x)

    x = Dropout(
        dropout_rate
    )(x)

    # Second causal dilated convolution
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)

    x = Dropout(
        dropout_rate
    )(x)

    # Match dimensions for residual connection
    if shortcut.shape[-1] != filters:

        shortcut = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
        )(shortcut)

    # Residual connection
    x = Add()([
        shortcut,
        x,
    ])

    x = Activation(
        "relu"
    )(x)

    return x


def build_tcn(
    lookback,
    number_of_features,
):
    """
    Build the Temporal Convolutional Network.
    """

    inputs = Input(
        shape=(
            lookback,
            number_of_features,
        ),
        name="tcn_input",
    )

    # --------------------------------------------------------
    # TCN residual blocks
    # --------------------------------------------------------

    x = residual_block(
        inputs,
        filters=64,
        kernel_size=3,
        dilation_rate=1,
    )

    x = residual_block(
        x,
        filters=64,
        kernel_size=3,
        dilation_rate=2,
    )

    x = residual_block(
        x,
        filters=64,
        kernel_size=3,
        dilation_rate=4,
    )

    x = residual_block(
        x,
        filters=64,
        kernel_size=3,
        dilation_rate=8,
    )

    # --------------------------------------------------------
    # Output head
    # --------------------------------------------------------

    x = GlobalAveragePooling1D()(
        x
    )

    x = Dense(
        64,
        activation="relu",
    )(x)

    x = Dropout(
        0.15
    )(x)

    outputs = Dense(
        1,
        activation="linear",
        name="solar_output",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="solar_tcn",
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=[
            "mae"
        ],
    )

    return model