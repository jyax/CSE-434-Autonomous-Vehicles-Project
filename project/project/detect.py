import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import numpy as np

# import model from roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="T1shm97ow35fVI0S8ksT")
project = rf.workspace().project("project-he2on")
model = project.version(3).model

class Detect(Node):
    def __init__(self):
        super().__init__('crosswalks')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.run_detect, 10)

    def run_detect(self, msg):
        # convert compressed image to cv2 image
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print(e)
            return

        # use roboflow model to detect crosswalks
        detections = model.predict(img, confidence=80, overlap=30).json()

        # draw bounding box
        imout = img.copy()
        for detection in detections['predictions']:
            x1 = int(detection['x'] - detection['width'] / 2)
            x2 = int(detection['x'] + detection['width'] / 2)
            y1 = int(detection['y'] - detection['height'] / 2)
            y2 = int(detection['y'] + detection['height'] / 2)
            print(f"{x1}, {x2}, {y1}, {y2}")
            cv2.rectangle(imout, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # display annotated result
        cv2.imshow('detection', imout)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = Detect()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

