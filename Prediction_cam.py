import dataset
import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import time

# Disable eager execution in TensorFlow 1.x
tf.compat.v1.disable_eager_execution()

# Initialize variables
last_checked_time = time.time()  # Track the last time the dataset was checked
video = cv2.VideoCapture(0)
time.sleep(2)  # Sleep for 2 seconds to let the camera warm up
CATEGORIES = ["cutting", "non cutting"]
print(CATEGORIES[0])
print(CATEGORIES[1])

# Path of training images
train_path = './data/train'

# Path of testing images
dir_path = './data/test'

# Check if the directories exist
if not os.path.exists(train_path):
    print("No such directory: train")
    raise Exception

if not os.path.exists(dir_path):
    print("No such directory: test")
    raise Exception

# Start the main loop
while True:
    # Capture frame-by-frame
    grabbed, frame = video.read()
    if not grabbed:
        break
    cv2.imshow("input", frame)

    # Wait for key press, if 'q' pressed, capture an image and process the dataset
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('./data/test/test.jpg', frame)
        cv2.waitKey(1)

        # Check if 30 seconds have passed since the last check
        current_time = time.time()
        if current_time - last_checked_time >= 30:  # 30 seconds interval
            last_checked_time = current_time  # Reset the last checked time

            # Iterate through test images and make predictions
            print("Checking dataset...")

            result_count = {"cutting": 0, "non cutting": 0}

            # Walk through all testing images one by one
            for root, dirs, files in os.walk(dir_path):
                for name in files:
                    image_path = os.path.join(root, name)
                    print("Processing:", image_path)

                    # Image preprocessing
                    image_size = 128
                    num_channels = 3
                    images = []

                    if os.path.exists(image_path):
                        image = cv2.imread(image_path)
                        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
                        images.append(image)
                        images = np.array(images, dtype=np.uint8)
                        images = images.astype('float32')
                        images = np.multiply(images, 1.0 / 255.0)

                        # Reshape image to fit the model input
                        x_batch = images.reshape(1, image_size, image_size, num_channels)

                        # Restore the model
                        sess = tf.Session()
                        saver = tf.train.import_meta_graph('model/trained_model.meta')
                        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

                        # Get default graph and tensors
                        graph = tf.get_default_graph()
                        y_pred = graph.get_tensor_by_name("y_pred:0")
                        x = graph.get_tensor_by_name("x:0")
                        y_true = graph.get_tensor_by_name("y_true:0")
                        y_test_images = np.zeros((1, len(os.listdir(train_path))))

                        # Feed the image data into the network
                        feed_dict_testing = {x: x_batch, y_true: y_test_images}
                        result = sess.run(y_pred, feed_dict=feed_dict_testing)

                        # Get predicted class with maximum probability
                        a = result[0].tolist()
                        max_prob = max(a)
                        index1 = a.index(max_prob)
                        predicted_class = CATEGORIES[index1]
                        print(f"Predicted class: {predicted_class} with confidence: {max_prob * 100}%")

                        # Update count of predictions
                        if predicted_class == "cutting":
                            result_count["cutting"] += 1
                        else:
                            result_count["non cutting"] += 1

            # Display the result count on the video frame
            cv2.putText(frame, f"Cutting: {result_count['cutting']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Non Cutting: {result_count['non cutting']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            # If less than 30 seconds passed, print waiting message
            print("Waiting for 30 seconds to check dataset...")

    # Display the frame with the count overlay
    cv2.imshow("input", frame)

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
