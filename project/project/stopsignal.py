import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

TIME_PERIOD = 10

class StopSignalPublisher(Node):

    def __init__(self):
        super().__init__('stopsignal_publisher')
        self.publisher_ = self.create_publisher(Bool, '/stopsignal', 10)
        timer_period = TIME_PERIOD # 10
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.stop = False # initially not stopped

    def timer_callback(self):
        msg = Bool()
        self.stop = not self.stop
        msg.data = self.stop
        self.publisher_.publish(msg)
        self.get_logger().info('Stop Signal: "%s"' % self.stop)


def main(args=None):
    rclpy.init(args=args)

    stopsignal_publisher = StopSignalPublisher()

    rclpy.spin(stopsignal_publisher)

if __name__ == '__main__':
    main()

