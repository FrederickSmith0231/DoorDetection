#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <detection_msgs/BoundingBoxes.h>
#include <librealsense2/rs.hpp> // Include the RealSense SDK
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h> // Add cv_bridge for image conversion
#include <image_transport/image_transport.h>
#include <geometry_msgs/Pose.h>

ros::Publisher goal_pose_publisher;

detection_msgs::BoundingBox door_bbox; // Global variable to store the current bounding box
detection_msgs::BoundingBox handle_bbox; // Global variable to store the current bounding box

bool doorDetected = false; // Global variable to indicate if a door was detected
bool handleFoundWithinDoor = false; // Global variable to indicate if a handle was detected within a door bounding box
bool handleIsRightSide = false; // Global variable to store which side of the door the handle is on

// Declare global variables for camera intrinsic parameters
double fx, fy, cx, cy;

float xmin, xmax, ymin, ymax, zmin, zmax;

cv::Point2f center_of_mass;  // Declare a global variable to store the center of mass

// Create a Pose message
geometry_msgs::Pose target_pose;

// Function to find a valid depth value in the vicinity of the given pixel coordinates
bool findValidDepthValue(cv_bridge::CvImageConstPtr cv_ptr, int& u, int& v, uint16_t& depth_value) {
    int search_radius = 5;  // Adjust the search radius as needed

    for (int r = 1; r <= search_radius; ++r) {
        for (int du = -r; du <= r; ++du) {
            for (int dv = -r; dv <= r; ++dv) {
                int u_neighbor = u + du;
                int v_neighbor = v + dv;

                // Check if the neighbor pixel is within the image bounds
                if (u_neighbor >= 0 && u_neighbor < cv_ptr->image.cols &&
                    v_neighbor >= 0 && v_neighbor < cv_ptr->image.rows) {
                    uint16_t neighbor_depth = cv_ptr->image.at<uint16_t>(v_neighbor, u_neighbor);

                    // Check if the neighbor depth value is valid (non-zero)
                    if (neighbor_depth != 0) {
                        u = u_neighbor;
                        v = v_neighbor;
                        depth_value = neighbor_depth;
                        return true;  // Valid depth value found
                    }
                }
            }
        }
    }

    return false;  // No valid depth value found in the vicinity
}

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
    // Extract the camera intrinsic parameters and assign them to global variables
    fx = camera_info_msg->K[0];
    fy = camera_info_msg->K[4];
    cx = camera_info_msg->K[2];
    cy = camera_info_msg->K[5];

    // // Optionally, print the parameters to verify
    // ROS_INFO("Camera Intrinsic Parameters:");
    // ROS_INFO("fx: %f", fx);
    // ROS_INFO("fy: %f", fy);
    // ROS_INFO("cx: %f", cx);
    // ROS_INFO("cy: %f", cy);
}

// Callback function to process YOLOv5 detection results
void yolov5DetectionCallback(const detection_msgs::BoundingBoxesConstPtr& detection_msg) {
    // Process YOLOv5 detection messages here

    // Assuming BoundingBoxes message contains a list of bounding boxes
    for (const auto& bbox : detection_msg->bounding_boxes) {
        detection_msgs::BoundingBox current_bbox = bbox; // Store the "Door" bounding box
        if (bbox.Class == "Door") {
            // If the current bounding box has Class "Door"
            // ROS_INFO("Door detected");
            doorDetected = true;

            // Now, let's look for a "Handle" bounding box within the current "Door" bounding box
            for (const auto& inner_bbox : detection_msg->bounding_boxes) {
                if (inner_bbox.Class == "Handle" &&
                    inner_bbox.xmin >= current_bbox.xmin && inner_bbox.xmax <= current_bbox.xmax &&
                    inner_bbox.ymin >= current_bbox.ymin && inner_bbox.ymax <= current_bbox.ymax) {
                    // If an inner bounding box has Class "Handle" and is within the "Door" bounding box
                    handle_bbox = inner_bbox; //assign to global
                    door_bbox = current_bbox; // assign to global 
                    handleFoundWithinDoor = true;
                    if ((current_bbox.xmin - inner_bbox.xmin) < (current_bbox.xmax - inner_bbox.xmax)) {
                        // Door is on the right side 
                        handleIsRightSide = true;
                    } else {
                        // Door is on the left side 
                        handleIsRightSide = false;
                    }

                } else {
                    handleFoundWithinDoor = false; // add extra condition to make this occur less
                }
            }
        } else if (bbox.Class == "Handle") {
            handle_bbox = current_bbox;
            handleFoundWithinDoor = false;
        } else {
            doorDetected = false;
        }
    }
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (handle_bbox.xmax > 0 | handle_bbox.ymax > 0 ) { //check to see if handle bbox is non zero 
        try {
            // Convert the ROS image message to an OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

            int64_t xmin = handle_bbox.xmin;
            int64_t ymin = handle_bbox.ymin;
            int64_t xmax = handle_bbox.xmax;
            int64_t ymax = handle_bbox.ymax;

            // Define the ROI based on the bounding box coordinates
            cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);

            // Extract the ROI from the image
            cv::Mat roi_image = cv_ptr->image(roi);

            // Display the ROI
            cv::imshow("ROI", roi_image);
            cv::waitKey(1);

            // Convert the ROI to grayscale
            cv::Mat grayImage;
            cv::cvtColor(roi_image, grayImage, CV_BGR2GRAY);

            // Display the grayscale ROI
            cv::imshow("Grayscale ROI", grayImage);
            cv::waitKey(1);

            // Apply Canny edge detection within the ROI
            cv::Mat edges;
            cv::Canny(grayImage, edges, 100, 200); // You can adjust the threshold values as needed

            // Display the edges within the ROI
            cv::imshow("Edges within ROI", edges);
            cv::waitKey(1);

            // Find the moments of the edges within the ROI
            cv::Moments mu = cv::moments(edges);

            // Calculate the center of mass within the ROI
            cv::Point2f cm(xmin + mu.m10 / mu.m00, ymin + mu.m01 / mu.m00);

            // Assign the computed center of mass to the global variable
            center_of_mass = cm;

            // Calculate the center of mass within the original image
            cv::Point2f cm_roi(cm.x - xmin, cm.y - ymin);

            // Draw a white circle at the center of mass location on the original colored image
            cv::circle(roi_image, cm_roi, 5, cv::Scalar(0, 0, 255), -1);
            cv::imshow("Center of Mass", roi_image); // Show the colored image with the white dot
            cv::waitKey(1);


            // Perform any other desired action
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }
}

void depthImageCallback(const sensor_msgs::ImageConstPtr& depth_image_msg) {
    try {
        // Convert ROS sensor_msgs::Image to OpenCV cv::Mat
        cv_bridge::CvImageConstPtr cv_ptr;
        cv_ptr = cv_bridge::toCvShare(depth_image_msg, sensor_msgs::image_encodings::TYPE_16UC1);

        // Check if the image is empty before proceeding
        if (cv_ptr->image.empty()) {
            ROS_WARN("Received an empty depth image.");
            return;
        }

        // Calculate the center of mass (CM) within the bounding box
        int u_cm = center_of_mass.x;
        int v_cm = center_of_mass.y;

        ROS_INFO("Center of mass: x %f, y %f pixels", center_of_mass.x, center_of_mass.y);

        // Extract the depth value at the center of mass location
        uint16_t depth_value_cm = cv_ptr->image.at<uint16_t>(v_cm, u_cm);

        // Attempt to find a valid depth value for the center of mass
        if (findValidDepthValue(cv_ptr, u_cm, v_cm, depth_value_cm)) {
            // Convert depth value to meters using camera intrinsic parameters
            float z_cm = depth_value_cm * 0.001;  // Convert depth from mm to meters

            // Use camera intrinsic parameters to compute X, Y, Z coordinates for the center of mass
            float x_cm = (u_cm - cx) * z_cm / fx;
            float y_cm = (v_cm - cy) * z_cm / fy;

            // Display depth value and 3D coordinates of the center of mass
            ROS_INFO("Depth Value at Center of Mass: %u mm", depth_value_cm);
            ROS_INFO("Center of Mass (X, Y, Z): (%f, %f, %f) meters", x_cm, y_cm, z_cm);

            // Store or use the depth and 3D coordinates of the center of mass as needed
            target_pose.position.x = x_cm;
            target_pose.position.y = y_cm;
            target_pose.position.z = z_cm;

            goal_pose_publisher.publish(target_pose);

            // // Create the annotated image by cloning the input depth image
            // cv::Mat annotated_image = cv_ptr->image.clone();

            // // Draw a white circle at the center of mass location
            // cv::circle(annotated_image, cv::Point(u_cm, v_cm), 5, cv::Scalar(255, 255, 255), -1);

            // // Display the annotated image
            // cv::imshow("Bounding Box on Depth Image", annotated_image);
            // cv::waitKey(1);
        } else {
            ROS_WARN("No valid depth value found for the center of mass in the vicinity.");
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("CV_Bridge Exception: %s", e.what());
    }
}

// void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
//     pcl::PCLPointCloud2 cloud_filtered;

//     if (doorDetected) {
//         // Convert to PCL data type
//         pcl::PCLPointCloud2 cloud;
//         pcl_conversions::toPCL(*cloud_msg, cloud);

//         // Create a pointer to hold the point cloud data
//         pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);

//         // Convert pcl::PCLPointCloud2 to pcl::PointCloud<pcl::PointXYZ>
//         pcl::fromPCLPointCloud2(cloud, *pcl_cloud_xyz);

//         // Print the number of points before filtering
//         ROS_INFO("Number of points before filtering: %lu", pcl_cloud_xyz->size());

//         // Create a PassThrough filter to extract points within the ROI
//         pcl::PassThrough<pcl::PointXYZ> pass;
//         pass.setInputCloud(pcl_cloud_xyz);
//         pass.setFilterFieldName("x");
//         pass.setFilterLimits(xmin, xmax);
//         pass.filter(*pcl_cloud_xyz);

//         // Print the number of points after voxelgrid filtering
//         ROS_INFO("Number of points after x pass through filtering: %lu", pcl_cloud_xyz->size());

//         pass.setInputCloud(pcl_cloud_xyz);
//         pass.setFilterFieldName("y");
//         pass.setFilterLimits(ymin, ymax);
//         pass.filter(*pcl_cloud_xyz);

//         // Print the number of points after voxelgrid filtering
//         ROS_INFO("Number of points after x and y pass through filtering: %lu", pcl_cloud_xyz->size());

//         // // Perform voxel grid filtering on the filtered point cloud
//         // pcl::VoxelGrid<pcl::PointXYZ> sor;
//         // sor.setInputCloud(pcl_cloud_xyz);
//         // sor.setLeafSize(0.1, 0.1, 0.1);
//         // sor.filter(*pcl_cloud_xyz);

//         // // Print the number of points after voxelgrid filtering
//         // ROS_INFO("Number of points after voxelgrid filtering: %lu", pcl_cloud_xyz->size());

//         // Convert pcl::PointCloud<pcl::PointXYZ> back to pcl::PCLPointCloud2
//         pcl::toPCLPointCloud2(*pcl_cloud_xyz, cloud_filtered);

//     } else {
//         // If doorDetected is false, create an empty point cloud
//         cloud_filtered.data.clear();
//         cloud_filtered.width = cloud_filtered.height = 0;
//         cloud_filtered.row_step = cloud_filtered.point_step = 0;
//         cloud_filtered.is_dense = false;
//     }

//     // Convert to ROS data type
//     sensor_msgs::PointCloud2 output;
//     pcl_conversions::fromPCL(cloud_filtered, output);

//     // Publish the filtered data (either filtered or empty)
//     point_cloud_publisher.publish(output);
// }

int main(int argc, char** argv) {
    ros::init(argc, argv, "yolov5_pointcloud_detection_node");
    ros::NodeHandle nh;

    // Subscribe to YOLOv5 detection messages
    ros::Subscriber yolov5_detection_sub = nh.subscribe("/yolov5/detections", 10, yolov5DetectionCallback);

    // Subscribe to D435 topic for camera color intrinsics 
    ros::Subscriber color_info_sub = nh.subscribe("/camera/aligned_depth_to_color/camera_info", 0.5, cameraInfoCallback);

    // Subscribe to the rgb image
    ros::Subscriber image_callback_sub = nh.subscribe("/camera/color/image_raw", 10, imageCallback);

    // Subscribe to the depth image
    ros::Subscriber depth_image_sub = nh.subscribe("/camera/aligned_depth_to_color/image_raw", 10, depthImageCallback);
    
    // // Subscribe to the point cloud
    // ros::Subscriber pointcloud_sub = nh.subscribe("/camera/depth/color/points", 10, pointCloudCallback);
    
    // // Advertise a publisher for the point cloud
    // point_cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/filtered_point_cloud", 1);

    // Advertise a publisher for the point cloud
    goal_pose_publisher = nh.advertise<geometry_msgs::Pose>("/goal_pose", 1);


    ros::spin(); // Keep the node running

    return 0;
}
    
