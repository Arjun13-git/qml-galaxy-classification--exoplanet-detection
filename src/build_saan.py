import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. The Front Door: Learnable Noise Gate ---
class LearnableNoiseGate(layers.Layer):
    def __init__(self, **kwargs):
        super(LearnableNoiseGate, self).__init__(**kwargs)

    def build(self, input_shape):
        # A tiny convolution to find noise patterns and create a 0-to-1 mask
        self.noise_evaluator = layers.Conv2D(
            filters=input_shape[-1], 
            kernel_size=(3, 3),      
            padding='same',
            activation='sigmoid',    
            name='noise_mask_generator'
        )
        super(LearnableNoiseGate, self).build(input_shape)

    def call(self, inputs):
        mask = self.noise_evaluator(inputs)
        return inputs * mask # Mutes the deep-space static

# --- 2. The Spotlight: Squeeze-and-Excitation Block ---
# --- 2. The Spotlight: Squeeze-and-Excitation Block ---
class SEBlock(layers.Layer):
    def __init__(self, channels, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        
        # 👇 ADD THESE TWO LINES 👇
        self.channels = channels
        self.ratio = ratio
        # ------------------------

        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation1 = layers.Dense(channels // ratio, activation='relu')
        self.excitation2 = layers.Dense(channels, activation='sigmoid')
        self.multiply = layers.Multiply()

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio,
        })
        return config
    
    def call(self, inputs):
        se = self.squeeze(inputs)
        se = self.excitation1(se)
        se = self.excitation2(se)
        
        se = tf.reshape(se, [-1, 1, 1, inputs.shape[-1]]) 
        return self.multiply([inputs, se])
# --- 3. The Main Engine: Assembling the SAAN ---
def build_saan_model(input_shape=(64, 64, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)
    
    # Pass raw image through the Noise Gate
    x = LearnableNoiseGate()(inputs)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = SEBlock(channels=32)(x) # Apply attention spotlight
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = SEBlock(channels=64)(x) # Apply attention spotlight
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = SEBlock(channels=128)(x) # Apply attention spotlight
    
    # The Head: Making the Final Decision
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Prevent overfitting
    
    # Output layer for 3 classes: Elliptical, Spiral, Irregular
    outputs = layers.Dense(num_classes, activation='softmax')(x) 
    
    model = models.Model(inputs, outputs, name="SAAN_Architecture")
    return model

# Quick test to see if it compiles
if __name__ == "__main__":
    model = build_saan_model()
    model.summary()