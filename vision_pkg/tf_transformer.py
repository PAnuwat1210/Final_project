import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

class TfTransformerNode(Node):
    def __init__(self):
        super().__init__('tf_transformer_node')
        
        # 1. ตั้งค่า TF2 Buffer และ Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 2. Subscribe รับพิกัด ArUco
        self.sub_pose = self.create_subscription(
            PoseStamped,
            '/aruco_pose',
            self.pose_callback,
            10
        )
        
        # 3. Publisher พิกัดเป้าหมายฝั่ง Dobot
        self.pub_dobot_pose = self.create_publisher(PoseStamped, '/dobot1_target_pose', 10)
        
        self.get_logger().info("TF2 Transformer Listener Node Started!")

    def pose_callback(self, msg):
        try:
            # Lookup Transformation จาก Frame ต้นทางไปยัง dobot1_base
            target_frame = 'dobot1_base'
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
                rclpy.time.Time()
            )
            
            # แปลงพิกัด โดยส่งเฉพาะ msg.pose (Pose) เข้าไปใน do_transform_pose
            transformed_pose_msg = do_transform_pose(msg.pose, transform)
            
            # ประกอบเป็น PoseStamped ใหม่เพื่อ Publish
            out_msg = PoseStamped()
            out_msg.header.stamp = msg.header.stamp
            out_msg.header.frame_id = target_frame
            out_msg.pose = transformed_pose_msg
            
            self.pub_dobot_pose.publish(out_msg)
            
            # ดึงตำแหน่ง position มาแสดงผล
            orig = msg.pose.position
            trans = transformed_pose_msg.position
            
            self.get_logger().info(
                f"\n[Camera Frame] X:{orig.x:.2f}, Y:{orig.y:.2f}, Z:{orig.z:.2f}\n"
                f"[Dobot1 Frame] X:{trans.x:.2f}, Y:{trans.y:.2f}, Z:{trans.z:.2f}"
            )
            
        except Exception as e:
            self.get_logger().warn(f"Transform failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = TfTransformerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()