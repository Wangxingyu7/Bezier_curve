import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.special import comb
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def draw_green_circles(image_path, output_dir, circle_positions, radius=40, color=(0, 255, 0), file_name='flaw_cheat.png'):
    """
    在图像上绘制绿色圆圈并保存
    :param image_path: 输入图像路径
    :param output_dir: 输出目录路径
    :param circle_positions: 圆圈的坐标列表，格式为 [(x1, y1), (x2, y2), ...]
    :param radius: 圆圈的半径
    :param color: 圆圈的颜色，格式为 (B, G, R)
    :param file_name: 输出文件名
    """
    img = cv2.imread(image_path)

    # 确保图像存在
    if img is None:
        print("Error: Could not load image.")
        return


    for (x, y) in circle_positions:
        cv2.circle(img, (x, y), radius, color, thickness=2)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)

    cv2.imwrite(output_path, img)

    cv2.imshow('Image with Green Circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    circle_positions = [(115, 260), (700, 260)]  # 根据实际情况调整坐标


    input_image_path = './image/gen_maze3.png'
    output_image_path = './image/'


    draw_green_circles(input_image_path, output_image_path, circle_positions)



def bezier_curve(control_points, t):
    n = len(control_points) - 1  # 控制点数量减1
    result = np.zeros_like(control_points[0], dtype=np.float64)  # 确保结果是浮点数
    for i, P in enumerate(control_points):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        result += P * bernstein
    return result

def convex_hull():
    control_points = np.array([
        [0, 0],
        [2, 4],
        [5, 3],
        [7, 5],
        [9, 2],
        [11, 3]
    ], dtype=np.float64)


    plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')


    t_values = np.linspace(0, 1, 100)
    bezier_points = np.array([bezier_curve(control_points, t) for t in t_values])

    plt.plot(bezier_points[:, 0], bezier_points[:, 1], label='Bezier Curve', color='blue')

    plt.plot(control_points[:, 0], control_points[:, 1], 'k--', label='Control Polygon')

    hull = ConvexHull(control_points)
    for simplex in hull.simplices:
        plt.plot(control_points[simplex, 0], control_points[simplex, 1], 'g--', label='Convex Hull' if simplex[0] == 0 else "")

    plt.legend()
    plt.title('Convex Hull of Bezier Curve')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def initial_path_example():
    fig, ax = plt.subplots()

    obs1 = patches.Polygon([[0.1, 0.8], [0.6, 0.8], [0.5, 0.5], [0.2, 0.6]], closed=True, color='gray', alpha=0.5)
    ax.add_patch(obs1)

    obs2 = patches.Polygon([[0.3, 0.5], [0.8, 0.4], [0.7, 0.2], [0.4, 0.3]], closed=True, color='gray', alpha=0.5)
    ax.add_patch(obs2)

    ax.plot(0.1, 0.2, 'ko', markersize=5)
    ax.plot(0.9, 0.8, 'ko', markersize=5)

    path = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.55], [0.5, 0.48], [0.6, 0.5], [0.7, 0.6], [0.8, 0.7], [0.9, 0.8]]
    ax.plot([x[0] for x in path], [x[1] for x in path], 'k-', lw=1)


    for point in path:
        ax.vlines(point[0], 0, 1, colors='gray', linewidth=1, linestyles='dotted')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True)

    plt.show()

def premature_convergence():

    fig, ax = plt.subplots()

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2 - Y**2) - 0.5 * np.exp(-((X-1)**2 + (Y+1)**2))

    contour = ax.contourf(X, Y, Z, levels=14, cmap='viridis')
    fig.colorbar(contour)

    local_min = (0, 0)
    global_min = (1, -1)
    ax.scatter(*local_min, color='red', s=100, label='local optimum')
    ax.scatter(*global_min, color='blue', s=100, label='global optimum')

    path_x = [0, 0.5, 0.6, 0.55, 0.52, 0.51, 0.505, 0.501]
    path_y = [0, -0.1, -0.15, -0.12, -0.11, -0.105, -0.101, -0.1005]
    ax.plot(path_x, path_y, 'ko-', label='搜索路径', markersize=5)

    ax.set_title('Example of early convergence')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)

    plt.show()

def instructure_pict_article():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.axis('off')

    boxes = [
        (0.2, 2.8, '开始'),
        (1.4, 2.8, '步骤1'),
        (2.6, 2.8, '步骤2'),
        (3.8, 2.8, '步骤3'),
        (5.0, 2.8, '步骤4'),

        (0.2, 1.6, '步骤5'),
        (1.4, 1.6, '步骤6'),
        (2.6, 1.6, '步骤7'),
        (3.8, 1.6, '步骤8'),
        (5.0, 1.6, '步骤9'),

        (0.2, 0.4, '步骤10'),
        (1.4, 0.4, '步骤11'),
        (2.6, 0.4, '步骤12'),
        (3.8, 0.4, '步骤13'),
        (5.0, 0.4, '结束')
    ]

    for x, y, text in boxes:
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        plt.text(x + 0.5, y + 0.5, text, fontsize=10, ha='center', va='center')

    for j in range(5):
        ax.annotate('', xy=(j * 1.2 + 0.6, 1.2), xytext=(j * 1.2 + 0.6, 2.2),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    for i in range(2):
        for j in range(5):
            ax.annotate('', xy=(j * 1.2 + 1.1, i * 1.2 + 1.1), xytext=(j * 1.2 + 0.1, i * 1.2 + 1.1),
                        arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.show()


if __name__ == "__main__":
    instructure_pict_article()