# Illegal-Tree-Logging-Detection-System
import os
import cv2
import time
import numpy as np
import tensorflow.compat.v1 as tf

# Disable eager execution in TensorFlow 1.x
tf.disable_eager_execution()

# Initialize variables
video = cv2.VideoCapture(0)
time.sleep(2)  # Let the camera warm up
CATEGORIES = ["cutting", "non cutting"]

# Paths for training and testing datasets
train_path = './data/train'
dir_path = './data/test'

# Check if the directories exist
if not os.path.exists(train_path):
    print("No such directory: train")
    raise Exception

if not os.path.exists(dir_path):
    print("No such directory: test")
    raise Exception

# Initialize timer and counter
last_checked_time = time.time()
check_count = 0
max_checks = 30

# Start the main loop
while check_count < max_checks:
    grabbed, frame = video.read()
    if not grabbed:
        print("Failed to grab frame.")
        break

    cv2.imshow("Input", frame)

    # Check for 'q' key to trigger processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save the frame as a test image
        cv2.imwrite('./data/test/test.jpg', frame)
        cv2.waitKey(1)

        current_time = time.time()
        if current_time - last_checked_time >= 30:  # 30 seconds interval
            last_checked_time = current_time  # Update the last checked time
            check_count += 1  # Increment check counter
            print(f"Check {check_count}/{max_checks}: Processing dataset...")

            result_count = {"cutting": 0, "non cutting": 0}

            # Walk through test images and predict
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

                        # Reshape image to fit model input
                        x_batch = images.reshape(1, image_size, image_size, num_channels)

                        # Restore the model
                        sess = tf.Session()
                        saver = tf.train.import_meta_graph('model/trained_model.meta')
                        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

                        # Get graph and tensors
                        graph = tf.get_default_graph()
                        y_pred = graph.get_tensor_by_name("y_pred:0")
                        x = graph.get_tensor_by_name("x:0")
                        y_true = graph.get_tensor_by_name("y_true:0")
                        y_test_images = np.zeros((1, len(CATEGORIES)))

                        # Feed the image data into the network
                        feed_dict_testing = {x: x_batch, y_true: y_test_images}
                        result = sess.run(y_pred, feed_dict=feed_dict_testing)

                        # Get predicted class with maximum probability
                        a = result[0].tolist()
                        max_prob = max(a)
                        index1 = a.index(max_prob)
                        predicted_class = CATEGORIES[index1]
                        print(f"Predicted class: {predicted_class} with confidence: {max_prob * 100:.2f}%")

                        # Update count of predictions
                        result_count[predicted_class] += 1

            # Overlay results on the frame
            cv2.putText(frame, f"Cutting: {result_count['cutting']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Non Cutting: {result_count['non cutting']}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the updated frame
            cv2.imshow("Input", frame)

        else:
            print("Waiting for 30 seconds before the next check...")

    # Display the frame
    cv2.imshow("Input", frame)

# Release resources
video.release()
cv2.destroyAllWindows()
print("Completed all checks.")
