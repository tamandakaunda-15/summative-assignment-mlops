
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Assuming constants are defined in preprocessing or passed in if needed
# from . import preprocessing

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 2
LEARNING_RATE = 0.0001

def create_model():
    print("Building the model...")
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                             include_top=False,
                             weights='imagenet')

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Model summary:")
    model.summary()
    return model

def train_model(model, train_gen, val_gen, epochs, models_path, model_filename):
    print("\nStarting the training process...")

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    checkpoint_filepath = os.path.join(models_path, model_filename)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        verbose=1
    )
    print("Training complete.")
    return history

def test_model(model, test_gen):
    print("Evaluating the model on the test set...")
    test_gen.reset()
    y_true = test_gen.classes

    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = tf.argmax(y_pred_probs, axis=1).numpy()

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"Test Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    return metrics

def retrain_model(model_path, train_gen, val_gen, learning_rate, new_epochs=5):
    print(f"Retraining model from {model_path} for {new_epochs} epochs...")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Cannot retrain.")
        return None

    model = tf.keras.models.load_model(model_path)
    base_model = model.layers[0]
    base_model.trainable = True

    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=learning_rate / 10),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_gen,
        epochs=new_epochs,
        validation_data=val_gen,
        verbose=1
    )
    print("\nModel retraining complete.")

    retrained_model_path = os.path.join(os.path.dirname(model_path), 'student_engagement_retrained.h5')
    model.save(retrained_model_path)
    print(f"Retrained model saved to {retrained_model_path}")

    return model
