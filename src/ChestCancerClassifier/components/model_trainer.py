import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from ChestCancerClassifier.entity.config_entity import (TrainingConfig)
from ChestCancerClassifier import logger


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        # Recompile the model with Adam optimizer for better convergence
        # Use learning rate from config
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),
            metrics=["accuracy"]
        )
        logger.info(f"Model compiled with Adam optimizer, learning_rate={self.config.params_learning_rate}")

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Setup callbacks for better training
        callbacks = [
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpointing to save best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        logger.info(f"Starting training with {self.config.params_epochs} epochs")
        logger.info(f"Training samples: {self.train_generator.samples}, Validation samples: {self.valid_generator.samples}")
        logger.info(f"Steps per epoch: {self.steps_per_epoch}, Validation steps: {self.validation_steps}")

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model (in case callbacks didn't save it)
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
        logger.info(f"Training completed. Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")