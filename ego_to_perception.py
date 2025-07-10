"""temp."""

import uuid

from autoware_perception_msgs.msg import (
    ObjectClassification,
    PredictedObject,
    PredictedObjectKinematics,
    PredictedObjects,
    PredictedPath,
    Shape,
)

from autoware_planning_msgs.msg import Trajectory

from builtin_interfaces.msg import Duration

from geometry_msgs.msg import (
    AccelWithCovarianceStamped,
    Point32,
    Polygon,
    Pose,
    Vector3,
)

from nav_msgs.msg import Odometry

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header

from tf_transformations import quaternion_matrix

from unique_identifier_msgs.msg import UUID

import yaml

def shift_pose_forward(pose: Pose, offset_x: float) -> Pose:
    """Return a copy of pose shifted forward along its local X-axis."""
    from tf_transformations import quaternion_matrix

    pose_shifted = Pose()
    pose_shifted.orientation = pose.orientation

    # Rotation matrix
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rot = quaternion_matrix(quat)[:3, :3]

    # Shift vector in local X
    offset_vector = np.array([offset_x, 0.0, 0.0])
    world_offset = rot @ offset_vector

    pose_shifted.position.x = pose.position.x + world_offset[0]
    pose_shifted.position.y = pose.position.y + world_offset[1]
    pose_shifted.position.z = pose.position.z + world_offset[2]

    return pose_shifted

def create_bounding_box_polygon(
    pose: Pose, base_to_front: float, base_to_rear: float, base_to_width: float
):
    """Create footprint polygon in map frame from vehicle dimensions and pose at base_link."""
    local_corners = np.array([
        [ base_to_front, -base_to_width, 0.0],
        [ base_to_front,  base_to_width, 0.0],
        [-base_to_rear,   base_to_width, 0.0],
        [-base_to_rear,  -base_to_width, 0.0],
    ])
    
    # Transform to world/map frame
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    trans = np.array([pose.position.x, pose.position.y, pose.position.z])
    rot_matrix = quaternion_matrix(quat)[:3, :3]

    world_corners = [trans + rot_matrix @ pt for pt in local_corners]

    polygon = Polygon()
    for pt in world_corners:
        polygon.points.append(Point32(x=pt[0], y=pt[1], z=pt[2]))
    polygon.points.append(polygon.points[0])  # Close the loop

    return polygon


def compute_dimensions(
    base_to_front: float, base_to_rear: float, base_to_width: float, height: float
) -> Vector3:
    """Compute bounding box dimensions."""
    dimensions = Vector3()
    dimensions.x = base_to_front + base_to_rear  # length
    dimensions.y = base_to_width * 2.0  # width
    dimensions.z = height  # height
    return dimensions


def to_duration(seconds: float) -> Duration:
    """Convert a float to a Duration message."""
    duration = Duration()
    duration.sec = int(seconds)
    duration.nanosec = int((seconds - duration.sec) * 1e9)
    return duration


def to_ros_uuid(u: uuid.UUID) -> UUID:
    """Convert a UUID to a ROS UUID message."""
    uuid_msg = UUID()
    uuid_msg.uuid = list(u.bytes)
    return uuid_msg


def make_car_classification() -> ObjectClassification:
    """Return ObjectClassification labeled as CAR with 100% confidence."""
    obj_class = ObjectClassification()
    obj_class.label = ObjectClassification.CAR
    obj_class.probability = 1.0
    return obj_class


def load_vehicle_info(path: str) -> dict:
    """Load vehicle parameters from a ROS 2-style YAML file with /**/ros__parameters."""
    with open(path, "r") as f:
        full_data = yaml.safe_load(f)
    return full_data.get("/**", {}).get("ros__parameters", {})


def get_geometry_params(vehicle_info: dict):
    """Compute geometry values needed to generate footprint and dimensions."""
    base_to_front = vehicle_info["wheel_base"] / 2 + vehicle_info["front_overhang"]
    base_to_rear = vehicle_info["wheel_base"] / 2 + vehicle_info["rear_overhang"]
    base_to_width = vehicle_info["wheel_tread"] / 2 + max(
        vehicle_info["left_overhang"], vehicle_info["right_overhang"]
    )
    height = vehicle_info["vehicle_height"]
    return base_to_front, base_to_rear, base_to_width, height


class TrajectoryListener(Node):
    """Node that subscribes to trajectory and localization topics."""

    def __init__(self):
        """Initialize subscriptions, publisher, and internal state."""
        super().__init__("trajectory_listener")

        self.predicted_object = PredictedObject()
        self.kinematics = PredictedObjectKinematics()
        self.object_id = UUID()

        static_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        self.object_id = to_ros_uuid(static_uuid)
        self.classification = make_car_classification()
        vehicle_info = load_vehicle_info('/home/zulfaqarazmi/autoware/pilot-auto.x2.gen2/src/description/vehicle/j6_gen2_description/j6_gen2_description/config/vehicle_info.param.yaml')

        self.base_to_front, self.base_to_rear, self.base_to_width, self.height = (
            get_geometry_params(vehicle_info)
        )
        self.dimensions = compute_dimensions(
            self.base_to_front, self.base_to_rear, self.base_to_width, self.height
        )

        # Publisher
        self.predicted_object_pub = self.create_publisher(
            PredictedObjects, "/debug/ego_as_predicted_object", 10
        )

        # Subscribers
        self.create_subscription(
            Trajectory,
            "/control/trajectory_follower/lateral/predicted_trajectory",
            self.trajectory_callback,
            10,
        )

        self.create_subscription(
            Odometry, "/localization/kinematic_state", self.odometry_callback, 10
        )

        self.create_subscription(
            AccelWithCovarianceStamped,
            '/localization/acceleration',
            self.acceleration_callback,
            10,
        )

    def trajectory_callback(self, msg: Trajectory):
        """Handle incoming trajectory messages."""
        if len(msg.points) < 2:
            return  # Not enough points to compute differences

        times = [
            pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            for pt in msg.points
        ]
        diffs = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]

        predicted_path = PredictedPath()
        self.kinematics.predicted_paths.clear()
        predicted_path.confidence = 1.0
        predicted_path.time_step = to_duration(sum(diffs) / len(diffs))
        predicted_path.path = [pt.pose for pt in msg.points]  # Use pose only

        self.kinematics.predicted_paths.append(predicted_path)

    def odometry_callback(self, msg: Odometry):
        """Handle incoming odometry messages."""
        self.kinematics.initial_pose_with_covariance = msg.pose
        self.kinematics.initial_twist_with_covariance = msg.twist

    def acceleration_callback(self, msg: AccelWithCovarianceStamped):
        """Handle incoming acceleration messages."""
        self.kinematics.initial_acceleration_with_covariance = msg.accel
        self.maybe_publish_predicted_object()

    def get_shape(self, pose: Pose) -> Shape:
        """Create and return the shape of the predicted object."""

        shape = Shape()
        polygon = create_bounding_box_polygon(
            pose,
            self.base_to_front,
            self.base_to_rear,
            self.base_to_width,
        )
        shape.footprint = polygon
        shape.type = Shape.BOUNDING_BOX
        shape.dimensions = self.dimensions
        return shape

    def maybe_publish_predicted_object(self):
        """Publish predicted object if all fields are populated."""
        self.predicted_object = PredictedObject()
        self.predicted_object.object_id = self.object_id
        self.predicted_object.classification.append(self.classification)
        self.predicted_object.existence_probability = 1.0
        self.predicted_object.kinematics = self.kinematics

        shifted_pose = shift_pose_forward(
            self.kinematics.initial_pose_with_covariance.pose,
            (self.base_to_front - self.base_to_rear) / 2.0)
        self.predicted_object.shape = self.get_shape(
            shifted_pose
        )
        predicted_objects = PredictedObjects()
        predicted_objects.header = Header(
            frame_id="map",
            stamp=self.get_clock().now().to_msg(),
        )
        predicted_objects.objects.append(self.predicted_object)
        self.predicted_object_pub.publish(predicted_objects)


def main(args=None):
    """Start the ROS2 node."""
    rclpy.init(args=args)
    node = TrajectoryListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
