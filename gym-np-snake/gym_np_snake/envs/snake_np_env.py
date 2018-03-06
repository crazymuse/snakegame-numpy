import numpy as np
import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from IPython.display import clear_output

class DisplayParam:
    emptyChar = ' '
    wallChar = '|'
    foodChar = 'x'
    snakeChar = 'o'
    gridtopleftChar = chr(0x250F)
    gridtoprightChar = chr(0x2513)
    gridbottomleftChar = chr(0x2517)
    gridbottomrightChar = chr(0x251B)
    gridverticalChar = chr(0x2503)
    gridhorizontalChar = chr(0x2501)
    arrowoffset = 0x2190
class Grid:
    def __init__(self, length, width):
        self.dp = DisplayParam()
        self.length, self.width = length, width
        self.grid = np.zeros((length, width),dtype=np.int32)
        self.grid[:] = ord(self.dp.emptyChar)
        
    def reset(self):
        self.grid = np.zeros((self.length,self.width),dtype=np.int32);
        self.grid[:]=ord(self.dp.emptyChar)
    def itemset(self,pos,val):
        self.grid[pos[0]][pos[1]] = val       

    def display(self):
        print (self.dp.gridtopleftChar+''.join([self.dp.gridhorizontalChar]*(self.width*2-1))+self.dp.gridtoprightChar)
        for row in self.grid:
            print(self.dp.gridverticalChar+' '.join([chr(r) for r in row])+self.dp.gridverticalChar)
        print (self.dp.gridbottomleftChar+''.join([self.dp.gridhorizontalChar]*(self.width*2-1))+self.dp.gridbottomrightChar)
        clear_output(wait=True)

class Snake:
    def __init__(self, pos, snake_id):
        self.length = 1
        self.head = pos
        self.snake_id = snake_id
        self.pos_ary = [pos]
        self.direction = np.random.randint(4)

    def move(self, action,food_pos):
        self.head = tuple(map(sum, zip(self.head, self.decodeAction(action)))) 
        eaten_something = False
        if(self.head == food_pos):
            eaten_something = True
        self.pos_ary.append(self.head)
        if not eaten_something:
            self.pos_ary.pop(0)
        else:
            self.length += 1
        return eaten_something;

    def reset(self,pos):
        self.length=1;
        self.head=pos;
        self.pos_ary=[pos];  


    def decodeAction(self,a):
        dirAry=[(0,-1),(-1,0),(0,1),(1,0)] #WNES
        if a == 1: # Left
            self.direction=(self.direction+4-1)%4
        elif a == 2:
            self.direction=(self.direction+1)%4
        
        return dirAry[self.direction]



class SnakeGame:
    def __init__(self,length,width,n_snakes=1):
        self.dp = DisplayParam()
        self.length,self.width,self.n_snakes = length,width,n_snakes;
        self.grid = Grid(length,width)
        self.n_snakes=n_snakes
        if n_snakes+1>=length*width:
            raise Exception('too many snakes')
        self.snakes=[]
        self.update_food()
        self.addSnakes()
        
    def has_snake(self,pos):
        for i in range(len(self.snakes)):
            if pos in self.snakes[i].pos_ary:
                return True;
        return False;

    def sample_empty_pos(self):
        pos = (np.random.randint(self.grid.length),np.random.randint(self.grid.width))
        while self.has_snake(pos) or pos == self.food_pos:
            pos = (np.random.randint(self.grid.length),np.random.randint(self.grid.width))
        return pos

    def addSnakes(self):
        pos_ary=[]
        for i in range(self.n_snakes):
            pos = (np.random.randint(self.grid.length),np.random.randint(self.grid.width))
            while pos in pos_ary or (pos  == self.food_pos):
                pos = (np.random.randint(self.grid.length),np.random.randint(self.grid.width))
            pos_ary.append(pos);
            self.snakes.append(Snake(pos,i))
            
    def inside_grid(self,pos):
        pos = list(pos)
        if pos[0]>=0 and pos[1]>=0 and pos[0]<self.grid.length and pos[1]<self.grid.width:
            return True
        return False

    def hasCollided(self):
        for snake in self.snakes:
            head, snake_id = snake.head, snake.snake_id; 
            if not self.inside_grid(head):
                return True
            for snake in self.snakes:
                if snake.snake_id==snake_id:
                    if head in snake.pos_ary[:-1]:
                        return True
                else:
                    if head in snake.pos_ary:
                        return True


    def update_food(self):
        food_pos = (np.random.randint(self.grid.length),np.random.randint(self.grid.width))
        while self.has_snake(food_pos):
            food_pos = (np.random.randint(self.grid.length),np.random.randint(self.grid.width))
        self.food_pos = food_pos



    def get_observation(self):
        obs = self.grid.grid.copy()
        obs[obs == ord(self.dp.emptyChar)] = 5
        obs[obs == ord(self.dp.foodChar)] = 255
        obs[obs == ord(self.dp.snakeChar)] = 200

        for i in range(4):
            obs[obs == self.dp.arrowoffset+i] = i*20+100

        return np.array(obs,dtype=np.uint8);

    def display(self,verbose=False):
        self.grid.reset()
        self.grid.itemset(self.food_pos,ord(self.dp.foodChar))
        for snake in self.snakes:
            for pos in snake.pos_ary:
                self.grid.itemset(pos,ord(self.dp.snakeChar))
            self.grid.itemset(snake.head, self.dp.arrowoffset+snake.direction)
                #self.grid.itemset(snake.head, '#')
        if verbose == True : 
            self.grid.display()
        return self.grid

    def get_state(self):
        states = np.zeros((self.n_snakes,4),dtype=np.float)
        for i in range(self.n_snakes):
            states[i][0] = float(self.snakes[i].direction)/4.0 # Direction of snake head
            states[i][1] = float(self.snakes[i].head[0])/self.grid.length
            states[i][2] = float(self.snakes[i].head[1])/self.grid.width
            states[i][3] = float(np.sum(np.absolute(np.array(self.snakes[i].head)-np.array(self.food_pos))))/(self.grid.length+self.grid.width)
        return states

    def reset(self):
        self.grid = Grid(self.length,self.width)
        if self.n_snakes+1>=self.length*self.width:
            raise Exception('too many snakes')
        self.snakes=[]
        self.update_food()
        self.addSnakes()
        return np.reshape(self.get_observation(),(1,self.length,self.width,1))
        

    def step(self,action_list,verbose=False):
        """
        state = 4 x nsnakes info with (direction,xpos,ypos,distance from food)
        """
        rewards,dones = np.zeros((self.n_snakes),dtype=np.float), np.zeros((self.n_snakes)).astype('bool')
        for i in range(self.n_snakes):
            eaten_food = self.snakes[i].move(action_list[i],self.food_pos)
            if eaten_food == True : 
                self.update_food()
                rewards[i]=1; # Eaten food 
            if self.hasCollided() == True:
                self.snakes[i].reset(self.sample_empty_pos())
                rewards[i]=-1;
                dones[i]=True;

        self.display(verbose=verbose)
        observation = self.get_observation();
        observation = np.repeat(observation,repeats=self.n_snakes,axis=0)
        states = self.get_state()
        return observation,rewards,dones,states;



    def close(self):
        pass


class HyperParameters:
    length = 100
    width = 100

class SnakeNpSingleEnv(gym.Env):
    """
    Gym wrapper over the numpy Game
    """
    def __init__(self):

        self.length=HyperParameters.length;
        self.width = HyperParameters.width;
        self.observation_space = spaces.Box(low=0,high=255,shape=(self.length, self.width, 1),dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.env = SnakeGame(self.length, self.width, 1)




    def step(self,action):
        ob, reward,episode_over, states = self.env.step([action])
        ob = np.reshape(ob,(self.length,self.width,1))
        reward = reward[0]
        episode_over = episode_over[0]
        info = {"states":states}
        return ob,reward,episode_over,info


    
    def reset(self):
        return self.env.reset()

    def render(self,mode='human',close=False,verbose=True):
        return self.env.display(verbose=verbose)

    def seed(self,seed_val):
        pass
