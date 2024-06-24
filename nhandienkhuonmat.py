import cv2
import numpy as np
import torch
import time
import pickle
from facenet_pytorch import MTCNN
from sort.sort import Sort
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class Person:
    def __init__(self, id, face):
        self.id = id
        self.face = face
        self.name = "Unknown"
        self.prob = 0.0
        self.timeout = 10
        self.show_face = None
        self.pre_name = "Unknown"
        self.show_prob = 0.0
        self.detections = []  # Store detection results
        self.first_detected_time = time.time()  # Timestamp when first detected

    def update(self, face, name="Unknown", prob=0.0):
        self.face = face
        self.timeout = 10
        self.name = name
        self.prob = prob

    def add_detection(self, name, prob):
        self.detections.append((name, prob))

    def finalize_name(self):
        if self.detections:
            self.name, self.prob = max(self.detections, key=lambda x: x[1])

class Detector:
    def __init__(self, min_width=0, list_len=5):
        self.stream = None
        self.list_len = list_len
        self.list_img_size = 0
        self.tl = (0, 0)
        self.br = (0, 0)
        self.scale = 1
        self.persons = {}

        # Load the face recognition model and label encoder
        logging.info("Loading Face Recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
        self.recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
        self.le = pickle.loads(open("output/le.pickle", "rb").read())

    def detect(self, stream_path, scale=1, roi=None):
        self.stream = cv2.VideoCapture(stream_path)
        ok, frame = self.stream.read()
        if not ok:
            return ok, None

        h, w = frame.shape[:2]
        self.list_img_size = h // self.list_len
        self.scale = scale

        # Define the ROI in the middle of the frame with full height
        if roi is None:
            roi_width = w // 2
            self.tl = (w // 4, 0)  # Full height
            self.br = (w // 4 + roi_width, h)
        else:
            self.tl, self.br = roi
            
        tracker = Sort(max_age=0, min_hits=3, iou_threshold=0.3)

        mtcnn = MTCNN(
            image_size=160,
            min_face_size=50,
            thresholds=[0.6, 0.7, 0.87],
            margin=20,
            post_process=False,
            device=device
        )

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ok, frame = self.stream.read()
                if not ok:
                    return ok, None

                frame_count += 1

                cropped_frame = frame[self.tl[1]:self.br[1], self.tl[0]:self.br[0]]
                rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                boxes, probs = mtcnn.detect(rgb)
                detections = np.empty((0, 5))
                if boxes is not None:
                    valid_idx = [True if prob > 0.99 and all(box > 0) else False
                                 for prob, box in zip(probs, boxes)]
                    boxes = boxes[valid_idx]
                    probs = probs[valid_idx]

                    detections = np.concatenate((boxes, probs.reshape(-1, 1)), axis=1)
                    self.track(tracker, frame, cropped_frame, self.tl, 1, detections=detections)

                self.draw_faces_on_frame(frame)

                cv2.rectangle(frame, pt1=tuple(self.tl), pt2=tuple(self.br), color=(0, 255, 0), thickness=2)
                yield ok, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            logging.info(f"FPS: {fps:.2f}")
            self.stream.release()

    def track(self, tracker, frame, cropped_frame, p1, x, detections=np.empty((0, 5))):
        tracked_objs = tracker.update(detections)
        dh, dw = cropped_frame.shape[:2]

        obj_list = []

        for boxes_with_ids in tracked_objs:
            x1, y1, x2, y2, obj_id = boxes_with_ids.astype(int)

            if x1 <= 0 or x2 >= dw or y1 <= 0 or y2 >= dh:
                continue

            face = crop((x1, y1, x2, y2), cropped_frame, padding=2)

            # Perform face recognition
            if obj_id not in self.persons.keys():
                self.persons[obj_id] = Person(obj_id, face)

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            self.embedder.setInput(faceBlob)
            vec = self.embedder.forward()
            preds = self.recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = self.le.classes_[j] if proba > 0.9 else "Unknown"

            # Update the person with recognition information
            current_person = self.persons[obj_id]
            current_person.add_detection(name, proba)
            if time.time() - current_person.first_detected_time > 0.5:
                current_person.finalize_name()

            x1, y1 = int(x1 / x) + p1[0], int(y1 / x) + p1[1]
            x2, y2 = int(x2 / x) + p1[0], int(y2 / x) + p1[1]

            tl = (x1, y1 > 25 and y1 - 25 or y1)
            br = (x2, y1 > 25 and y1 or y1 + 25)

            cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

            obj_list.append(obj_id)

            cv2.rectangle(frame, pt1=tl, pt2=br, color=(0, 255, 0), thickness=1)
            cv2.putText(
                frame,
                f"{current_person.name}  {current_person.prob:.2f}",
                org=(x1 + 5, tl[1] + 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 150, 255),
                thickness=2
            )

        if len(self.persons.keys()) > self.list_len:
            tmp = sorted(self.persons.keys(), key=lambda x: self.persons[x].timeout)
            tmp = tmp[0]
            del self.persons[tmp]

        for id_ in list(self.persons.keys()):
            if id_ not in obj_list:
                if self.persons[id_].timeout < 1:
                    del self.persons[id_]
                else:
                    self.persons[id_].timeout -= 1

    def draw_faces_on_frame(self, frame):
        i = 0
        for index, id_ in enumerate(self.persons.keys()):
            index -= i
            if self.persons[id_].show_face is not None and self.persons[id_].pre_name != "Unknown":
                img_ = np.zeros((self.list_img_size, self.list_img_size, 3), dtype=np.uint8)
                face = cv2.resize(self.persons[id_].show_face, (self.list_img_size, self.list_img_size))

                tl_list = (index * self.list_img_size, 0)
                br_list = (index * self.list_img_size + self.list_img_size, self.list_img_size)

                frame[tl_list[0]:br_list[0], tl_list[1]:br_list[1]] = face

                cv2.putText(
                    frame,
                    f"{self.persons[id_].pre_name}",
                    (10, br_list[0] - 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2
                )
            else:
                i += 1

def resize(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def crop(box, frame, padding=0):
    return frame[max(0, box[1] - padding):min(box[3] + padding, frame.shape[0]),
                 max(0, box[0] - padding):min(box[2] + padding, frame.shape[1])]

def main():
    stream_path = "IMG_9393.mp4"
    # output_path = "demo.avi" # Un-comment this code snipset to saved the video
    detector = Detector()
    detect = detector.detect(stream_path=stream_path, scale=1, roi=None)

    # Read the first frame to get the original dimensions
    ok, frame = next(detect)
    if not ok:
        logging.error("Failed to read the video.")
        return

    # Get the frame size
    frame_height, frame_width = frame.shape[:2]

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    # out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    # Set the window size to match the original frame size
    cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detector", frame.shape[1], frame.shape[0])

    for ok, frame in detect:
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detector", frame)

        # Write the frame to the output video
        # out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release everything if job is finished
    # out.release() # Saved the video
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
