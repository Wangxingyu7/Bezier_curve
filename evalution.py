import numpy as np
import matplotlib.pyplot as plt
import process_image
import bezier_curve
import os
from PIL import Image

def euclidean_distance(path):
    """
    计算路径的欧式距离
    :param path: 路径点数组，形状为 (n, 2)
    :return: 欧式距离
    """
    distance = 0
    for i in range(len(path) - 1):
        distance += np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
    return distance

def obstacle_intersection_count(curve, obstacle_mask):
    """
    计算曲线与障碍物相交的次数
    :param curve: 曲线点数组，形状为 (n, 2)
    :param obstacle_mask: 障碍物掩码
    :return: 相交次数
    """
    intersect_count = 0
    for point in curve:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < obstacle_mask.shape[1] and 0 <= y < obstacle_mask.shape[0]:
            if obstacle_mask[y, x]:
                intersect_count += 1
    return intersect_count

def extract_path_from_image(image_path, color):
    """
    从图像中提取指定颜色的路径
    :param image_path: 图像路径
    :param color: 路径颜色，如 'b' 表示蓝色，'r' 表示红色
    :return: 路径点数组
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    if color == 'b':
        # 提取蓝色路径（A* 初始路径）
        path_indices = np.where((img_array[:, :, 2] > 200) & (img_array[:, :, 0] < 100) & (img_array[:, :, 1] < 100))
    elif color == 'r':
        # 提取红色路径（贝塞尔曲线）
        path_indices = np.where((img_array[:, :, 0] > 200) & (img_array[:, :, 1] < 100) & (img_array[:, :, 2] < 100))
    else:
        raise ValueError("Unsupported color. Use 'b' for blue or 'r' for red.")
    path = np.column_stack((path_indices[1], path_indices[0]))
    return path

def evaluate_paths(input_image_path, gen_image_path):
    # 获取障碍物掩码
    obstacle_mask = process_image.get_obstacle_mask_hsv(input_image_path)
    if obstacle_mask is None:
        return

    # 从生成的图像中提取路径信息
    initial_path = extract_path_from_image(gen_image_path, 'b')
    traj = extract_path_from_image(gen_image_path, 'r')

    # 计算欧式距离
    a_star_distance = euclidean_distance(initial_path)
    bezier_distance = euclidean_distance(traj)

    # 计算障碍物相交次数
    a_star_intersection_count = obstacle_intersection_count(initial_path, obstacle_mask)
    bezier_intersection_count = obstacle_intersection_count(traj, obstacle_mask)

    print(f"A* Path Euclidean Distance: {a_star_distance}")
    print(f"Bezier Curve Euclidean Distance: {bezier_distance}")
    print(f"A* Path Obstacle Intersection Count: {a_star_intersection_count}")
    print(f"Bezier Curve Obstacle Intersection Count: {bezier_intersection_count}")

    # 绘制评价结果图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 欧式距离比较
    labels = ['A* Path', 'Bezier Curve']
    distances = [a_star_distance, bezier_distance]
    axes[0].bar(labels, distances)
    axes[0].set_title('Euclidean Distance Comparison')
    axes[0].set_ylabel('Distance')

    # 障碍物相交次数比较
    intersection_counts = [a_star_intersection_count, bezier_intersection_count]
    axes[1].bar(labels, intersection_counts)
    axes[1].set_title('Obstacle Intersection Count Comparison')
    axes[1].set_ylabel('Intersection Count')

    # 生成保存路径
    base_filename = os.path.basename(input_image_path)  # 获取原文件名
    output_filename = os.path.join("image", f"eval_{base_filename}")  # 生成新的文件名

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    input_image_path = input("Please enter the original image file name (e.g., maze1.png): ")
    input_image_path = './image/' + input_image_path  # 拼接成完整的路径
    gen_image_path = input("Please enter the generated image file name (e.g., gen_maze1.png): ")
    gen_image_path = './image/' + gen_image_path  # 拼接成完整的路径

    evaluate_paths(input_image_path, gen_image_path)