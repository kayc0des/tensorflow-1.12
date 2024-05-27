## Sections:

### Importing Libraries:

The script starts by importing necessary libraries including TensorFlow, NumPy, and Matplotlib.
Creating Placeholders:

- The create_placeholders function creates TensorFlow placeholders for input data (x) and labels (y).
Creating Layers:

- The create_layer function creates a dense layer in the neural network using TensorFlow's tf.layers.Dense module.
Forward Propagation:

- The forward_prop function builds the forward propagation graph of the neural network using the provided layer sizes and activation functions.
Calculating Accuracy:

-  The calculate_accuracy function computes the accuracy of the model's predictions using TensorFlow operations.
Calculating Loss:

- The calculate_loss function computes the loss of the model's predictions using TensorFlow operations.
Creating Training Operation:

- The create_train_op function creates the training operation for the neural network using TensorFlow's gradient descent optimizer.
Training the Model:

- The train function builds, trains, and saves the neural network classifier using TensorFlow. It prints training and validation metrics during training.
Evaluating the Model:

- The evaluate function evaluates the output of the trained neural network using TensorFlow. It returns the prediction, accuracy, and loss.
### One-Hot Encoding:

The one_hot function converts an array to a one-hot matrix, which is commonly used in classification tasks.
### Main Section:

The main section of the script loads the MNIST dataset, evaluates the trained model on the test set, and visualizes a sample of predictions using Matplotlib.

## TensorFlow (`tf`) Functions:

#### 1. `tf.placeholder`:
- **Purpose**: Creates a placeholder for input data or labels.
- **Usage**: `tf.placeholder(dtype, shape, name)`
  - `dtype`: Data type of the placeholder.
  - `shape`: Shape of the placeholder.
  - `name`: Optional name for the placeholder.
- **Description**: Placeholders are used to feed actual data into the TensorFlow computational graph during the execution phase.

#### 2. `tf.layers.Dense`:
- **Purpose**: Creates a fully connected layer.
- **Usage**: `tf.layers.Dense(units, activation, kernel_initializer, name)`
  - `units`: Number of neurons in the layer.
  - `activation`: Activation function to be applied to the output.
  - `kernel_initializer`: Initializer for the weight matrix.
  - `name`: Optional name for the layer.
- **Description**: The Dense layer is a standard fully connected layer where each neuron is connected to every neuron in the previous and next layers.

#### 3. `tf.contrib.layers.variance_scaling_initializer`:
- **Purpose**: Initializes weights using a variance scaling initializer.
- **Usage**: `tf.contrib.layers.variance_scaling_initializer(mode)`
  - `mode`: Mode of the variance scaling initializer (`"FAN_IN"`, `"FAN_OUT"`, or `"FAN_AVG"`).
- **Description**: This initializer sets the standard deviation of the weights based on the fan-in, fan-out, or average of the fan-in and fan-out of the layer.

#### 4. `tf.reduce_mean`:
- **Purpose**: Computes the mean of elements across dimensions of a tensor.
- **Usage**: `tf.reduce_mean(input_tensor, axis=None, keepdims=False)`
  - `input_tensor`: The tensor to reduce.
  - `axis`: The dimensions to reduce.
  - `keepdims`: If true, retains reduced dimensions with length 1.
- **Description**: Reduces the input tensor along the specified dimensions by computing the mean value.

#### 5. `tf.cast`:
- **Purpose**: Casts a tensor to a new data type.
- **Usage**: `tf.cast(x, dtype)`
  - `x`: The tensor to cast.
  - `dtype`: The data type to cast to.
- **Description**: Converts the data type of the input tensor to the specified data type.

#### 6. `tf.equal`:
- **Purpose**: Computes element-wise equality between two tensors.
- **Usage**: `tf.equal(x, y)`
  - `x`, `y`: Tensors to compare for equality.
- **Description**: Returns a tensor of the same shape as the inputs with True values where x and y are equal, and False otherwise.

#### 7. `tf.argmax`:
- **Purpose**: Finds the indices of the maximum value along a specified axis.
- **Usage**: `tf.argmax(input, axis=None)`
  - `input`: The input tensor.
  - `axis`: The axis along which to find the maximum values.
- **Description**: Returns the index with the largest value across axes of a tensor.

#### 8. `tf.train.GradientDescentOptimizer`:
- **Purpose**: Optimizes the model using the gradient descent algorithm.
- **Usage**: `tf.train.GradientDescentOptimizer(learning_rate)`
  - `learning_rate`: The learning rate for the gradient descent algorithm.
- **Description**: This optimizer applies the gradient descent optimization algorithm to minimize the loss function.

#### 9. `tf.train.Saver`:
- **Purpose**: Saves and restores TensorFlow models.
- **Usage**: `tf.train.Saver()`
- **Description**: The Saver class provides methods for saving and restoring models, allowing you to save trained models to disk and later restore them for further use or evaluation.

#### 10. `tf.Session`:
- **Purpose**: Creates a TensorFlow session to execute operations.
- **Usage**: `with tf.Session() as sess:`
- **Description**: Sessions encapsulate the environment in which Operation objects are executed and Tensor objects are evaluated.

#### 11. `tf.global_variables_initializer`:
- **Purpose**: Initializes all global variables in the graph.
- **Usage**: `tf.global_variables_initializer()`
- **Description**: This initializer initializes all variables in the TensorFlow graph. It is commonly used before training a model to ensure all variables are properly initialized.

#### 12. `tf.train.import_meta_graph`:
- **Purpose**: Imports the TensorFlow graph from a meta-graph file.
- **Usage**: `tf.train.import_meta_graph(meta_graph_or_file)`
- **Description**: This function imports the TensorFlow graph structure saved in a meta-graph file, allowing you to load a pre-trained model's graph definition.

#### 13. `tf.train.GradientDescentOptimizer.minimize`:
- **Purpose**: Minimizes a loss function using the gradient descent algorithm.
- **Usage**: `tf.train.GradientDescentOptimizer.minimize(loss)`
  - `loss`: The loss tensor to minimize.
- **Description**: This method applies the gradient descent optimization algorithm to minimize the provided loss tensor.

