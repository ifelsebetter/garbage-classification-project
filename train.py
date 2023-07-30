import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_data = 'dataset/train'
validation_data = 'dataset/valid'

num_classes = 5

input_size = (224, 224)

batch_size = 32
epochs = 15

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


train_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

validation_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(
    train_data,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_data_generator.flow_from_directory(
    validation_data,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(input_size[0], input_size[1], 3),
    weights='imagenet'
)

model.trainable = True
for layer in model.layers[:-10]:
    layer.trainable = False

flatten = tf.keras.layers.Flatten()(model.output)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(flatten)
model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model_checkpoint = ModelCheckpoint('garbage_classification_model.h5', monitor='val_accuracy', save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

model.save('garbage_classification_model.h5')

loss, accuracy = model.evaluate(validation_generator)
print(f'Validation loss: {loss:.4f}')
print(f'Validation accuracy: {accuracy:.4f}')
