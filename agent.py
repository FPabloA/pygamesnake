import torch
import random
import numpy as np
from game import SnakeGameAI, Point, Direction
from collections import deque
from model import Linear_QNet, QTrainer
#from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9  #discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        #get points surrounding head
        pt_u = Point(head.x, head.y - 20)
        pt_d = Point(head.x, head.y + 20)
        pt_l = Point(head.x - 20, head.y)
        pt_r = Point(head.x + 20, head.y)

        dir_U = game.direction == Direction.UP
        dir_D = game.direction == Direction.DOWN
        dir_L = game.direction == Direction.LEFT
        dir_R = game.direction == Direction.RIGHT

        state = [
            #straight is a collision
            (dir_U and game.is_collision(pt_u)) or 
             (dir_D and game.is_collision(pt_d)) or 
             (dir_L and game.is_collision(pt_l)) or
             (dir_R and game.is_collision(pt_r)),

            #right is a collision
            (dir_U and game.is_collision(pt_r)) or 
             (dir_D and game.is_collision(pt_l)) or 
             (dir_L and game.is_collision(pt_u)) or
             (dir_R and game.is_collision(pt_d)),
            
            #left is a collision
            (dir_U and game.is_collision(pt_l)) or 
             (dir_D and game.is_collision(pt_r)) or 
             (dir_L and game.is_collision(pt_d)) or
             (dir_R and game.is_collision(pt_u)),

             dir_U, dir_D, dir_L, dir_R,

             game.food.y < game.head.y, #food up
             game.food.y > game.head.y, #food down
             game.food.x < game.head.x, #food left
             game.food.x > game.head.x  #food right
        ]
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory)> BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print(f"Game: {agent.n_games} Score: {score} Record: {record}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()