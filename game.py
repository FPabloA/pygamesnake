import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
pygame.font.init()
font = pygame.font.SysFont('arial', 25)
#font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('point', 'x, y')

WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(w//2, h//2)
        self.snake = [Point(self.head.x, self.head.y),
                      Point(self.head.x - BLOCK_SIZE, self.head.y + BLOCK_SIZE), 
                      Point(self.head.x - BLOCK_SIZE*2, self.head.y + BLOCK_SIZE*2)]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0


    def place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake:
            self.place_food()
    
    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.move(action)
        self.snake.insert(0, self.head)
        game_over = False
        reward = 0
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward -= 10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward += 10
            self.place_food()
        else:
            self.snake.pop()
        self.update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def is_collision(self, pt = None):
        if not pt:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

    def update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()

    def move(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)
        new_dir = clockwise(idx)
        if np.array_equal(action, [1,0,0]):
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0,1,0]):
            new_dir = clockwise[(idx+1)%4]
        elif np.array_equal(action, [0,0,1]):
            new_dir = clockwise[(idx-1)%4]
            
        self.direction = new_dir
        x,y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)

    
