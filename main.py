import numpy as np
import matplotlib.pyplot as plt
import bezier_curve
import process_image
import cv2
import os


def main():
    input_image_path = input("Please enter the image file name (e.g., maze1.png): ")
    input_image_path = './image/' + input_image_path

    clicked_points = process_image.process_image(input_image_path)
    if len(clicked_points) < 2:
        print("Error: At least two points are needed to generate a curve.")
        return

    start = tuple(clicked_points[0])
    end = tuple(clicked_points[1])

    original_obstacle_mask = process_image.get_obstacle_mask_hsv(input_image_path)
    if original_obstacle_mask is None:
        return

    min_gap = process_image.get_min_obstacle_gap(original_obstacle_mask, default_min_gap=5)
    kernel_size = int(np.ceil(min_gap))
    kernel_size = max(kernel_size, 10)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_obstacle_mask = cv2.dilate(original_obstacle_mask, kernel, iterations=1)

    dilated_free_space = ~dilated_obstacle_mask
    dilated_initial_path = process_image.generate_initial_path(start, end, dilated_free_space)

    if dilated_initial_path is None:
        print("No path found from start to goal using dilated obstacle mask.")
        return

    original_free_space = ~original_obstacle_mask
    original_initial_path = process_image.generate_initial_path(start, end, original_free_space)

    if original_initial_path is None:
        print("No path found from start to goal using original obstacle mask.")
        return

    smooth_path = bezier_curve.smooth_path_with_bspline(np.array(dilated_initial_path), degree=3, num_points=500)
    traj = bezier_curve.generate_bezier_curve_segments(smooth_path, n_points=500)

    if bezier_curve.check_curve_with_obstacles(traj, original_obstacle_mask):
        print("Generated curve intersects with obstacles. Adjusting control points...")
        smooth_path = bezier_curve.adjust_control_points(smooth_path, original_obstacle_mask)
        traj = bezier_curve.generate_bezier_curve_segments(smooth_path, n_points=500)

    # 生成并保存包含三条曲线及其欧式距离的左右布局图
    base_filename = os.path.basename(input_image_path)
    output_filename = os.path.join("image", f"gen_{base_filename}")
    bezier_curve.plot_curve_with_distance(input_image_path, original_initial_path, dilated_initial_path, traj, dilated_obstacle_mask, output_filename)

    # 生成并保存贝塞尔曲线及其曲率变化的左右布局图
    curvature_output_filename = os.path.join("image", f"cur_{base_filename}")
    bezier_curve.plot_curve_and_curvature(input_image_path, traj, curvature_output_filename)

if __name__ == "__main__":
    main()