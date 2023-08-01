import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("garbage_classification_model.h5")
class_labels = ['glass', 'organic_waste', 'plastic']
bin_colors = {
    'glass': (255, 255, 0),
    'organic_waste': (0, 255, 0),
    'plastic': (0, 0, 255)
}

def initialize_object_tracker(frame, bbox):
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bbox)
    return tracker

def update_object_tracker(frame, tracker):
    success, bbox = tracker.update(frame)
    return success, bbox

def detect_object(frame):
    image = cv2.resize(frame, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    predictions = model.predict(image)
    class_index = np.argmax(predictions[0])
    label = class_labels[class_index]

    return label

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    object_label = detect_object(frame)

    if object_label != 'background':
        if 'tracker' not in locals():
            bbox = (100, 100, 200, 200)
            tracker = initialize_object_tracker(frame, bbox)
        else:
            # Update object tracker
            success, bbox = update_object_tracker(frame, tracker)

            if success:
                bbox = tuple(map(int, bbox))
            else:
                del tracker
                bbox = (100, 100, 200, 200)
                tracker = initialize_object_tracker(frame, bbox)

        x, y, w, h = bbox

        color = bin_colors[object_label]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.putText(frame, object_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
