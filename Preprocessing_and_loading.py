import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'C:/Users/cemai/PycharmProjects/Handwriting_Neural_Network/grayscale_images/trainset/lines'
test_dir = 'C:/Users/cemai/PycharmProjects/Handwriting_Neural_Network/grayscale_images/testset/lines'

# Set batch size and image dimension
batch_size = 32
image_size = (128, 128)

# Create an ImageDataGenerator for data preprocessing
datagen = ImageDataGenerator(rescale=1./255)
# Load and preprocess the training dataset
train_dataset = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Load and preprocess the test dataset
test_dataset = datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Print the class labels
print("Class Labels:", test_dataset.class_indices)

# Print the number of classes
num_classes = len(test_dataset.class_indices)
print("Number of Classes:", num_classes)

# Print the shape of the input data
input_shape = test_dataset.image_shape
print("Input Shape:", input_shape)

# Now you can proceed with training your neural network using the loaded and preprocessed data
