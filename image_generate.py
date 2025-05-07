import random
from PIL import Image
import os

# 迷宫单元格类
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
        self.visited = False

    def remove_wall(self, other):
        if self.x < other.x:
            self.walls['right'] = False
            other.walls['left'] = False
        elif self.x > other.x:
            self.walls['left'] = False
            other.walls['right'] = False
        elif self.y < other.y:
            self.walls['bottom'] = False
            other.walls['top'] = False
        elif self.y > other.y:
            self.walls['top'] = False
            other.walls['bottom'] = False


# 迷宫类
class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[Cell(x, y) for y in range(height)] for x in range(width)]
        self.generate()

    def get_cell(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[x][y]
        return None

    def get_unvisited_neighbors(self, cell):
        neighbors = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for dx, dy in directions:
            neighbor = self.get_cell(cell.x + dx, cell.y + dy)
            if neighbor and not neighbor.visited:
                neighbors.append(neighbor)
        return neighbors

    def generate(self):
        stack = []
        current = self.cells[0][0]
        current.visited = True
        stack.append(current)

        while stack:
            neighbors = self.get_unvisited_neighbors(current)
            if neighbors:
                next_cell = random.choice(neighbors)
                current.remove_wall(next_cell)
                next_cell.visited = True
                stack.append(next_cell)
                current = next_cell
            else:
                current = stack.pop()

    def save_image(self, filename):
        cell_size = 20
        img_width = self.width * cell_size + 1
        img_height = self.height * cell_size + 1
        img = Image.new('RGB', (img_width, img_height), color='white')
        pixels = img.load()

        wall_color = (0, 0, 0)  # 黑色墙壁
        for x in range(self.width):
            for y in range(self.height):
                cell = self.cells[x][y]
                if cell.walls['top']:
                    for i in range(cell_size + 1):
                        pixels[x * cell_size + i, y * cell_size] = wall_color
                if cell.walls['right']:
                    for i in range(cell_size + 1):
                        pixels[(x + 1) * cell_size, y * cell_size + i] = wall_color
                if cell.walls['bottom']:
                    for i in range(cell_size + 1):
                        pixels[x * cell_size + i, (y + 1) * cell_size] = wall_color
                if cell.walls['left']:
                    for i in range(cell_size + 1):
                        pixels[x * cell_size, y * cell_size + i] = wall_color

        img.save(filename)


if __name__ == "__main__":
    if not os.path.exists('image'):
        os.makedirs('image')
    existing_files = [f for f in os.listdir('image') if f.startswith('maze') and f.endswith('.png')]
    numbers = [int(f[4:-4]) for f in existing_files if f[4:-4].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    filename = f'image/maze{next_number}.png'
    maze = Maze(10, 10)
    maze.save_image(filename)