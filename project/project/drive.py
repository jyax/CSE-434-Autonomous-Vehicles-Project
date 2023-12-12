import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

REGULAR_SPEED = 1.0
SLOW_SPEED = 0.3

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

    """
    Returns True if detects crosswalk
    """
    def crosswalk_cb(self, msg):
        self.detect_crosswalk = msg.data
        self.process_cb()

    """
    Returns True if detects stop signal
    """
    def signal_cb(self, msg):
        self.detect_signal = msg.data
        self.process_cb()

    """
    Drives the robot
    """
    def process_cb(self):
        twist_msg = Twist()
        twist_msg.linear.x = REGULAR_SPEED # 1.0

        if self.detect_crosswalk:
            if self.detect_signal:
                twist_msg.linear.x = 0.0 # stop
            else:
                twist_msg.linear.x = SLOW_SPEED # 0.3
        self.get_logger().info('Driving with velocity %f' % twist_msg.linear.x)
        self.publisher_.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)

    drive_node = Drive()

    rclpy.spin(drive_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    drive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
