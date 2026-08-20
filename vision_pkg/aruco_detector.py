import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.bridge = CvBridge()
        
        # 1. สร้าง Publisher สำหรับส่งพิกัด 3D ออกไปใช้งานต่อ
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)
        
        # 2. Subscribe ข้อมูล CameraInfo เพื่อดึงค่า Focal Length (fx, fy) และ Principal Point (cx, cy)
        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None

        # 3. Synchronize ภาพ Color และ Depth ให้เข้าพร้อมกัน
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.image_callback)
        
        # 4. ตั้งค่า ArUco Dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()

        self.get_logger().info("ArUco 3D Pose Detector Node Started!")

    def info_callback(self, msg):
        # อ่านค่า Intrinsic Matrix จากกล้อง
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.ppx = msg.k[2]
        self.ppy = msg.k[5]

    def image_callback(self, color_msg, depth_msg):
        try:
            # แปลงภาพจาก ROS Message เป็น OpenCV Format
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # ตรวจจับ Marker
            if hasattr(cv2.aruco, 'ArucoDetector'):
                detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                corners, ids, rejected = detector.detectMarkers(cv_color)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(cv_color, self.dictionary, parameters=self.parameters)
            
            if ids is not None and self.fx is not None:
                for i, corner in enumerate(corners):
                    pts = corner[0]
                    u = int(np.mean(pts[:, 0]))  # พิกัด pixel x
                    v = int(np.mean(pts[:, 1]))  # พิกัด pixel y
                    
                    # อ่านค่าความลึก Depth (หน่วยมิลลิเมตร)
                    depth_mm = cv_depth[v, u]
                    
                    if depth_mm > 0:
                        # แปลงเป็นหน่วยเมตร
                        z_c = depth_mm / 1000.0
                        
                        # คำนวณพิกัด X_c, Y_c (หน่วยเมตร) ด้วยสูตร Pinhole Camera Model
                        x_c = (u - self.ppx) * z_c / self.fx
                        y_c = (v - self.ppy) * z_c / self.fy
                        
                        # วาดกรอบและแสดงค่า 3D บนภาพ
                        cv2.polylines(cv_color, [np.int32(pts)], True, (0, 255, 0), 2)
                        cv2.circle(cv_color, (u, v), 5, (0, 0, 255), -1)
                        
                        text = f"ID:{ids[i][0]} XYZ:({x_c:.2f}, {y_c:.2f}, {z_c:.2f})m"
                        cv2.putText(cv_color, text, (u - 60, v - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Publish ค่าพิกัดออกทาง ROS 2 Topic
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = color_msg.header.stamp
                        pose_msg.header.frame_id = color_msg.header.frame_id
                        pose_msg.pose.position.x = float(x_c)
                        pose_msg.pose.position.y = float(y_c)
                        pose_msg.pose.position.z = float(z_c)
                        self.pose_pub.publish(pose_msg)
                        
                        self.get_logger().info(f"Marker ID {ids[i][0]} -> X: {x_c:.3f}m, Y: {y_c:.3f}m, Z: {z_c:.3f}m")

            cv2.imshow("RealSense ArUco 3D Detection", cv_color)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()