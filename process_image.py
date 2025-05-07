import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import heapq


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Trapezoid:
    def __init__(self, top, bottom, left_point, right_point):
        self.top = top
        self.bottom = bottom
        self.left_point = left_point
        self.right_point = right_point

    def __repr__(self):
        return f"Trapezoid(top={self.top}, bottom={self.bottom}, left={self.left_point}, right={self.right_point})"


class RoadMap:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, point):
        self.nodes.append(point)

    def add_edge(self, point1, point2):
        self.edges.append((point1, point2))


def process_image(input_image_path):
    '''
    处理图像，点击生成起点和终点
    :param input_image_path:
    :return:
    '''
    img = Image.open(input_image_path)
    img_array = np.array(img)

    fig, ax = plt.subplots()
    ax.imshow(img_array)
    ax.set_title('Click to set start (blue) and end (green) points. Right-click to finish.')
    ax.set_xlim(0, img_array.shape[1])
    ax.set_ylim(img_array.shape[0], 0)

    clicked_points = []

    width, height = img_array.shape[1], img_array.shape[0]
    min_dim = min(width, height)
    radius = max(1, min_dim // 100)

    def onclick(event):
        nonlocal clicked_points
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        clicked_points.append((int(x), int(y)))
        if len(clicked_points) == 1:
            draw_circle(int(x), int(y), [0, 0, 255])
        elif len(clicked_points) == 2:
            draw_circle(int(x), int(y), [0, 255, 0])
            plt.close()
        ax.imshow(img_array)
        plt.draw()

    def on_right_click(event):
        if event.button == 3 and len(clicked_points) >= 2:
            plt.close()

    def draw_circle(x, y, color):
        for i in range(max(0, int(y - radius)), min(height, int(y + radius + 1))):
            for j in range(max(0, int(x - radius)), min(width, int(x + radius + 1))):
                dx = j - x
                dy = i - y
                if dx * dx + dy * dy <= radius * radius:
                    img_array[i, j] = color

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('button_press_event', on_right_click)

    plt.show()
    return clicked_points


def get_obstacle_mask_hsv(input_image_path):
    '''
    定义障碍物掩码
    :param input_image_path:
    :return:
    '''
    img = cv2.imread(input_image_path)
    if img is None:
        print("Error: Could not load image.")
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([100, 100, 100])
    obstacle_mask = cv2.inRange(hsv, lower_black, upper_black)
    return obstacle_mask


def is_in_free_space(point, free_space):
    '''
    判读是否处于自由区域
    :param point:
    :param free_space:
    :return:
    '''
    x, y = point
    if 0 <= x < free_space.shape[1] and 0 <= y < free_space.shape[0]:
        return free_space[int(y), int(x)]
    return False


def heuristic(a, b):
    '''
    兼顾八个方向的启发式函数
    :param a:
    :param b:
    :return:
    '''
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


def a_star(start, goal, free_space):
    '''
    A star算法生成初始折线
    :param start:
    :param goal:
    :param free_space:
    :return:
    '''
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        # 8个方向的邻居点
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角线方向
        ]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if not is_in_free_space(neighbor, free_space):
                continue
            tentative_g_score = g_score[current] + 1  # 对角线方向的代价可以调整为根号2
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def generate_initial_path(start, goal, free_space):
    '''
    返回一个列表path,元素为各中间点
    '''
    path = a_star(start, goal, free_space)
    if path is None:
        print("No path found from start to goal.")
    return path

def get_min_obstacle_gap(obstacle_mask, default_min_gap=5):
    """
    获取图像中障碍物之间的最小间隙
    :param obstacle_mask: 障碍物掩码
    :param default_min_gap: 如果没有找到有效的间隙，则返回的默认最小间隙大小
    :return: 障碍物之间的最小间隙大小
    """
    # 计算距离变换，得到每个像素点到最近障碍物的距离
    distance_map = cv2.distanceTransform(~obstacle_mask, cv2.DIST_L2, 3)
    # 找到最小的非零距离作为障碍物之间的最小间隙
    min_gap = np.min(distance_map[distance_map > 0]) if np.any(distance_map > 0) else default_min_gap
    return max(min_gap, default_min_gap)

