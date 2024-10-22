Project Report: Speaker Recognition with Residual Neural Network

Introduction
Speaker recognition, a subfield of audio signal processing and machine learning, involves
identifying and verifying individuals based on their unique voice characteristics. This project
aims to implement a speaker recognition system using a Convolutional Neural Network
(CNN) architecture with residual blocks. The dataset used for training and validation consists
of audio recordings of various speakers. The trained model can predict and recognize
speakers' identities from input audio samples.

Data Collection and Preprocessing
The project begins by collecting an audio dataset, which is organized into two main folders:
'audio' containing speech recordings and 'noise' containing background noise samples. The
dataset is pre-processed by adjusting the sampling rate to 16,000 Hz and segregating
speakers' folders from 'other' and '_background_noise_' folders. The noise samples are also
processed to ensure a consistent sampling rate.

import tensorflow as tf
import os
from os.path import isfile, join
import numpy as np
import shutil
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio
import subprocess
# Copy data from Google Drive to local directory
!cp -r "/content/drive/MyDrive/16000_pcm_speeches" ./
data_directory = "./16000_pcm_speeches"
audio_folder = "audio"
noise_folder = "noise"
audio_path = os.path.join(data_directory, audio_folder)
noise_path = os.path.join(data_directory, noise_folder)
# Display the audio path
print(audio_path)
# Valid split ratio for train-validation split
valid_split = 0.1

Name : T. Venkata Bhuvan
Roll No. : 20B21A43A2
Branch : CAI

# Seed for shuffling data
shuffle_seed = 43
# Sample rate for audio
sample_rate = 16000
# Scaling factor for noise augmentation
scale = 0.5
# Batch size for training
batch_size = 128
# Number of training epochs
epochs = 15
# Move folders to appropriate paths (audio or noise)
for folder in os.listdir(data_directory):
if os.path.isdir(os.path.join(data_directory, folder)):
if folder in [audio_folder, noise_folder]:
continue
elif folder in ["other", "_background_noise_"]:
print("Moving folder to noise path:", folder)
shutil.move(
os.path.join(data_directory, folder),
os.path.join(noise_path, folder),
)
else:
print("Moving folder to audio path:", folder)
shutil.move(
os.path.join(data_directory, folder),
os.path.join(audio_path, folder),
)
In this code section:

1. Copies the 16000_pcm_speeches directory from Google Drive to the local working directory.
2. Defines paths for the audio and noise folders within the data directory.
3. Sets parameters such as valid split ratio, shuffle seed, sample rate, scaling factor, batch size,
and number of epochs.
4. Moves folders within the data directory to appropriate paths (audio or noise) based on their
names.

Data Augmentation
To enhance model generalization and robustness, noise is added to the audio samples at
varying scales using random background noise from the 'noise' folder. This augmentation aids
the model in learning to distinguish speakers' unique characteristics despite varying acoustic
environments.

# Load noise samples and prepare for augmentation
noise_paths = []
for subdir in os.listdir(noise_path):
subdir_path = Path(noise_path) / subdir
if os.path.isdir(subdir_path):
noise_paths += [
os.path.join(subdir_path, filepath)
for filepath in os.listdir(subdir_path)
if filepath.endswith(".wav")
]
# Command to adjust sample rate of noise files to 16000 Hz
command = (
"for dir in `ls -1 " + noise_path + "`; do "
"for file in `ls -1 " + noise_path + "/$dir/*.wav`; do "
"sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
"$file | grep sample_rate | cut -f2 -d=`; "
"if [ $sample_rate -ne 16000 ]; then "
"ffmpeg -hide_banner -loglevel panic -y "
"-i $file -ar 16000 temp.wav; "
"mv temp.wav $file; "
"fi; done; done"
)
# Execute the command to adjust sample rate of noise files
os.system(command)
# Load a noise sample
def load_noise_sample(path):
sample, sampling_rate = tf.audio.decode_wav(
tf.io.read_file(path), desired_channels=1
)
if sampling_rate == sample_rate:
slices = int(sample.shape[0] / sample_rate)
sample = tf.split(sample[: slices * sample_rate], slices)
return sample
else:
print("Sampling rate for", path, "is incorrect")
return None
# Load and stack noise samples
noises = []
for path in noise_paths:
sample = load_noise_sample(path)
if sample:
noises.extend(sample)
noises = tf.stack(noises)

# Function to add noise to audio samples
def add_noise(audio, noises=None, scale=0.5):
if noises is not None:
tf_rnd = tf.random.uniform(
(tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
)
noise = tf.gather(noises, tf_rnd, axis=0)
prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)
audio = audio + noise * prop * scale
return audio

In this code section:

1. Loads noise samples from the specified noise paths.
2. Checks and adjusts the sample rate of noise files to 16000 Hz using FFmpeg.
3. Defines a function load_noise_sample to load and split noise samples.
4. Loads and stacks the noise samples using the defined function.
5. Defines a function add_noise to add noise to audio samples. This function takes audio
samples, randomly selects noise samples, adjusts their amplitude, and adds them to the audio
samples.

Data Conversion and Transformation

The audio data is converted to spectrogram-like representations by performing the Short-
Time Fourier Transform (STFT) to transform the audio waveforms into a frequency domain.

This transformation facilitates the neural network in extracting meaningful features from the
audio samples.

import tensorflow as tf
# Function to convert file paths and labels to a dataset
def paths_and_labels_to_dataset(audio_paths, labels):
path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
audio_ds = path_ds.map(lambda x: path_to_audio(x))
label_ds = tf.data.Dataset.from_tensor_slices(labels)
return tf.data.Dataset.zip((audio_ds, label_ds))
# Function to read audio file and decode into audio tensor
def path_to_audio(path):
audio = tf.io.read_file(path)
audio, _ = tf.audio.decode_wav(audio, 1, sample_rate)
return audio

# Function to transform audio to its FFT (spectrogram) representation
def audio_to_fft(audio):
audio = tf.squeeze(audio, axis=-1)
fft = tf.signal.fft(
tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
)
fft = tf.expand_dims(fft, axis=-1)
return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])
# Create class names based on the directories in audio path
class_names = os.listdir(audio_path)
print(class_names)
# Collect audio file paths and labels
audio_paths = []
labels = []
for label, name in enumerate(class_names):
print("Speaker:", name)
dir_path = Path(audio_path) / name
speaker_sample_paths = [
os.path.join(dir_path, filepath)
for filepath in os.listdir(dir_path)
if filepath.endswith(".wav")
]
audio_paths += speaker_sample_paths
labels += [label] * len(speaker_sample_paths)
# Shuffle audio_paths and labels using the same seed
rng = np.random.RandomState(shuffle_seed)
rng.shuffle(audio_paths)
rng = np.random.RandomState(shuffle_seed)
rng.shuffle(labels)
# Calculate the number of validation samples
num_val_samples = int(valid_split * len(audio_paths))
# Split data into training and validation sets
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]
# Create datasets for training and validation
train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=batch_size * 8, seed=shuffle_seed).batch(
batch_size
)
valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)

valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=shuffle_seed).batch(32)
# Apply noise augmentation to training dataset
train_ds = train_ds.map(
lambda x, y: (add_noise(x, noises, scale=scale), y),
num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
# Transform audio samples to FFT representations
train_ds = train_ds.map(
lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.map(
lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

In this code section :

• The paths_and_labels_to_dataset function creates a dataset from audio file paths and their
corresponding labels.
• The path_to_audio function reads an audio file, decodes it into an audio tensor, and
resamples it to the specified sample rate.
• The audio_to_fft function transforms audio samples into their FFT (spectrogram)
representations, capturing their frequency domain features.
• Class names are generated based on the directories in the audio path.
• Audio file paths and labels are collected for creating the datasets.
• The data is split into training and validation sets, and noise augmentation is applied to the
training dataset using the add_noise function.
• Audio samples are transformed into FFT representations for both the training and
validation datasets.

Model Architecture
The core of the speaker recognition system is a Convolutional Neural Network (CNN) model
with residual blocks. Residual blocks enhance the training process and enable the network to
learn complex patterns effectively. The model consists of multiple residual blocks with
varying numbers of convolutional layers and pooling operations, followed by fully connected
layers for classification.

from tensorflow.keras.layers import Conv1D

def residual_block(x, filters, conv_num=3, activation="relu"):
s = keras.layers.Conv1D(filters, 1, padding="same")(x)
for i in range(conv_num - 1):
x = keras.layers.Conv1D(filters, 3, padding="same")(x)
x = keras.layers.Activation(activation)(x)
x = keras.layers.Conv1D(filters, 3, padding="same")(x)
x = keras.layers.Add()([x, s])
x = keras.layers.Activation(activation)(x)
return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
def build_model(input_shape, num_classes):
inputs = keras.layers.Input(shape=input_shape, name="input")
x = residual_block(inputs, 16, 2)
x = residual_block(x, 32, 2)
x = residual_block(x, 64, 3)
x = residual_block(x, 128, 3)
x = residual_block(x, 128, 3)
x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
return keras.models.Model(inputs=inputs, outputs=outputs)
# Build the model
model = build_model((sample_rate // 2, 1), len(class_names))
# Display model summary
model.summary()
# Compile the model
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy",
metrics=["accuracy"])
In this code section:

• The residual_block function defines a residual block with convolutional layers. This
block helps capture hierarchical features from the audio spectrogram.
• The build_model function constructs the overall model architecture. It stacks multiple
residual blocks with increasing numbers of filters and convolutional layers. The
architecture is designed to capture important patterns in the audio data.

• The model is built using the build_model function, with input shape (sample_rate // 2,
1) (half of the sample rate, as only the positive frequencies are considered) and output
classes equal to the number of class names (speakers).
• The model summary is displayed to provide an overview of the model architecture.
• The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss
function, and accuracy as the evaluation metric.

Training and Evaluation
The model is trained using the pre-processed and augmented dataset. The training is carried
out for a specified number of epochs with early stopping to prevent overfitting. The model is
compiled with the Adam optimizer and sparse categorical cross-entropy loss function.
Training progress and performance are monitored using validation accuracy and loss metrics.

# Define the filename for saving the best model checkpoint
model_save_filename = "model.h5"
# Define early stopping and model checkpoint callbacks
earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename,
monitor="val_accuracy", save_best_only=True)
# Train the model
history = model.fit(
train_ds,
epochs=epochs,
validation_data=valid_ds,
callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)
# Evaluate the model on the validation dataset
eval_result = model.evaluate(valid_ds)
print("Accuracy of model:", eval_result)
In this code section :
• The model_save_filename is defined to specify the filename for saving the best model
checkpoint during training.
• The earlystopping_cb callback is created to implement early stopping based on the
validation accuracy. It helps prevent overfitting by stopping training when the model's
performance on the validation set does not improve.
• The mdlcheckpoint_cb callback is used to save the best model checkpoint based on
validation accuracy.
• The model.fit function is called to train the model. The training loop runs for the
specified number of epochs and includes the training and validation datasets. The
earlystopping_cb and mdlcheckpoint_cb callbacks are applied during training.

Testing and Predictions
The trained model is evaluated using a separate test dataset. The system's predictions are
displayed alongside the true speaker identities, providing insights into the model's accuracy.
Additionally, the project demonstrates a prediction function that can take external audio
samples and predict the speaker's identity.

# Number of samples to display during testing
SAMPLES_TO_DISPLAY = 10
# Create a test dataset from validation data
test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
test_ds = test_ds.shuffle(buffer_size=batch_size * 8, seed=shuffle_seed).batch(
batch_size
)
# Apply noise augmentation to test dataset
test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, scale=scale), y))
# Initialize variables for storing predictions
true_labels = []
predicted_labels = []
# Iterate through test dataset and make predictions
for audios, labels in test_ds.take(1):
ffts = audio_to_fft(audios)
y_pred = model.predict(ffts)
rnd = np.random.randint(0, batch_size, SAMPLES_TO_DISPLAY)
audios = audios.numpy()[rnd, :, :]
labels = labels.numpy()[rnd]
y_pred = np.argmax(y_pred, axis=-1)[rnd]
for index in range(SAMPLES_TO_DISPLAY):
true_label = class_names[labels[index]]
predicted_label = class_names[y_pred[index]]
true_labels.append(true_label)
predicted_labels.append(predicted_label)
print("True Label:", true_label)
print("Predicted Label:", predicted_label)
print("Welcome" if true_label == predicted_label else "Sorry")
# Print overall accuracy of the model
accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
print("Overall Accuracy:", accuracy)

In this code section :
• The SAMPLES_TO_DISPLAY variable defines the number of samples to display
during testing and prediction.
• A test dataset is created from the validation data, and noise augmentation is applied to
the dataset.
• The loop iterates through the test dataset, converts audio samples to FFT
representations, makes predictions using the trained model, and displays the true and
predicted labels for a random subset of samples.
• The true and predicted labels are stored in lists for further analysis.
• The overall accuracy of the model is calculated by comparing the true and predicted
labels.

Results and Discussion
The accuracy of the speaker recognition model on the validation dataset is reported,
providing insight into its performance. The model's ability to correctly identify speakers is
showcased through test predictions. Any discrepancies between predicted and true identities
are highlighted, demonstrating the model's capabilities and potential areas for improvement.

Conclusion
The project successfully implements a speaker recognition system using a CNN architecture
with residual blocks. The system demonstrates accurate speaker identification and can be
extended for real-world applications, such as voice authentication and security systems.
Further enhancements could include exploring different CNN architectures, optimizing
hyperparameters, and incorporating additional data augmentation techniques to enhance the
model's robustness.
