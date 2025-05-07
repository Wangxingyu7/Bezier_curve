import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_hsv_values(input_image_path):
    # 加载图像
    img = cv2.imread(input_image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # 将图像转换为HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 显示图像
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Click on the image to see HSV values. Close the window to exit.')
    plt.axis('off')

    # 获取鼠标点击事件
    def on_click(event):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            print(f"Position: ({x}, {y})")
            print(f"Hue: {hsv[y, x, 0]}, Saturation: {hsv[y, x, 1]}, Value: {hsv[y, x, 2]}")

    # 连接点击事件
    plt.connect('button_press_event', on_click)

    plt.show()

if __name__ == "__main__":
    input_image_path = 'image/input_image.jpg'  # 输入图像路径
    show_hsv_values(input_image_path)