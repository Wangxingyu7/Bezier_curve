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


def convex_example_1():
    """
    绘制一个直角拐角和一条避开拐角的曲线，并在曲线上标记五个红色点。
    左边图：绘制所有红色点生成的凸包（绿色）。
    右边图：只用中间4个点绘制凸包。
    """

    def draw_corner(ax):
        ax.plot([0.2, 0.2], [0.2, 0.7], 'k', linewidth=2)
        ax.plot([0.2, 0.7], [0.7, 0.7], 'k', linewidth=2)

    def draw_curve(ax):
        t = np.linspace(0.4 * np.pi, np.pi, 1000)
        x = 0.5 * np.cos(t) + 0.5
        y = 0.5 * np.sin(t) + 0.5
        ax.plot(x, y, 'b', linewidth=2)
        return x, y, t

    def draw_convex_hull(ax, points, color='g', linewidth=2):
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], color=color, linewidth=linewidth)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect(1)
    ax1.axis('off')
    ax1.set_title("Before Encrypted Control Point", fontsize=12)

    draw_corner(ax1)
    x, y, t = draw_curve(ax1)
    points = np.array([[x[i], y[i]] for i in [0, 250, 500, 750, -1]])  # 选择曲线上的5个点
    for point in points:
        ax1.plot(point[0], point[1], 'ro', markersize=6)
    draw_convex_hull(ax1, points, color='g', linewidth=2)

    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_aspect(1)
    ax2.axis('off')
    ax2.set_title("After Encrypted Control Point", fontsize=12)

    draw_corner(ax2)
    x, y, t = draw_curve(ax2)
    new_points_indices = [200, 300, 400, 500, 600, 700]  # 选择曲线上的6个新点
    new_points = np.array([[x[i], y[i]] for i in new_points_indices])
    for point in new_points:
        ax2.plot(point[0], point[1], 'ro', markersize=6)
    draw_convex_hull(ax2, new_points, color='g', linewidth=2)

    plt.rcParams.update({
        'font.size': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.linewidth': 1.5,
        'grid.linewidth': 1.0,
        'grid.linestyle': '--',
        'grid.color': '#cccccc',
    })

    plt.show()



if __name__ == "__main__":
    convex_example_1()