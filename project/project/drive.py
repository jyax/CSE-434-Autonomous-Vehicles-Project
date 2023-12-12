import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class Drive(Node):
    def __init__(self):
        super().__init__('drive')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.detect_crosswalk = False
        self.detect_signal = False

        self.crosswalk_sub = self.create_subscription(
            Bool,
            '/crosswalk',
            self.crosswalk_cb,
            10)

        self.signal_sub = self.create_subscription(
            Bool,
            '/stopsignal',
            self.signal_cb,
            10)

    def crosswalk_cb(self, msg):
        self.detect_crosswalk = msg.data
        self.process_cb()

    def signal_cb(self, msg):
        self.detect_signal = msg.data
        self.process_cb()

    def process_cb(self):
        twist_msg = Twist()
        twist_msg.angular.z = 0.0
        twist_msg.linear.x = 0.6 

        if self.detect_crosswalk:
            if self.detect_signal:
                twist_msg.linear.x = 0.0
            else:
                twist_msg.linear.x = 0.2

        self.publisher_.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)

    drive_node = Drive()

    rclpy.spin(drive_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
