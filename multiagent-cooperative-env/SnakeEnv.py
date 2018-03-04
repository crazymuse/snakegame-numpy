import numpy as np;


class Grid:
    def __init__(self, length, width):
        self.length, self.width = length, width
        self.grid = np.chararray((length, width))
        self.grid[:] = '.'
        
    def reset(self):
        self.grid = np.chararray((self.length,self.width));
        self.grid[:]='.'
    def itemset(self,pos,val):
        self.grid.itemset(pos,val)        

    def display(self):
        for row in self.grid:
            print('|'+' '.join(row.decode('utf-8'))+'|')


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
        dirAry=[(-1,0),(0,1),(1,0),(0,-1)] #NESW
        if a == 1: # Left
            self.direction=(self.direction+4-1)%4
        elif a == 2:
            self.direction=(self.direction+1)%4
        
        return dirAry[self.direction]



class SnakeGame:
    def __init__(self,length,width,n_snakes=1):
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
        obs = np.zeros((1,self.grid.length,self.grid.width,1),dtype=float)
        for x in range(self.grid.length):
            for y in range(self.grid.width):
                obs[0][x][y][0] = int(ord(self.grid.grid[x][y]))
        obs[obs == ord('.')] = 5
        obs[obs == ord('x')] = 255

        return np.array(obs,dtype=np.uint8);

    def display(self,verbose=False):
        self.grid.reset()
        self.grid.itemset(self.food_pos,'x')
        for snake in self.snakes:
            for pos in snake.pos_ary:
                self.grid.itemset(pos,'o')
            self.grid.itemset(snake.head, chr(48+snake.snake_id))
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


    def step(self,action_list,verbose=False):
        """
        state = 4 x nsnakes info with (direction,xpos,ypos,distance from food)
        """
        rewards,dones = np.zeros((self.n_snakes),dtype=np.float), np.zeros((self.n_snakes)).astype('bool')
        for i in range(self.n_snakes):
            eaten_food = self.snakes[i].move(action_list[i],self.food_pos)
            if eaten_food == True : 
                self.update_food()
                rewards[i]=0.9; # Eaten food 
            elif self.hasCollided() == True:
                self.snakes[i].reset(self.sample_empty_pos())
                rewards[i]=-0.2;
                dones[i]=True;
            else:
                rewards[i] = 0.1*(1.0-float(np.sum(np.absolute(np.array(self.snakes[i].head)-np.array(self.food_pos))))/(self.grid.length+self.grid.width))
                if action_list[i] != 0:
                    rewards[i] = rewards[i] - 0.1

        self.display(verbose=verbose)
        observation = self.get_observation();
        observation = np.repeat(observation,repeats=self.n_snakes,axis=0)
        states = self.get_state()
        return observation,rewards,dones,states;



    def close(self):
        pass

class MultiAgentSnakeGame:
    def __init__(self,length,width,n_envs=1,n_snakes=1):
        self.n_envs = n_envs;
        self.n_snakes = n_snakes;
        self.s_games = [ SnakeGame(length,width,n_snakes=n_snakes) for i in range(n_envs)]
    
    def step(self,action_list,verbose = False):
        action_list = action_list.reshape(self.n_envs,self.n_snakes)
        m_obs,m_rewards,m_dones,m_states = [],[],[],[]
        for i in range(self.n_envs):
            observation,rewards,dones,states = self.s_games[i].step(action_list[i])
            for j in range(self.n_snakes):
                m_obs.append(observation[j])
                m_rewards.append(rewards[j])
                m_dones.append(dones[j])
                m_states.append(states[j])
        if verbose == True:
            self.display(verbose=verbose)       
        m_obs,m_rewards,m_dones,m_states = np.array(m_obs),np.array(m_rewards),np.array(m_dones),np.array(m_states)
        return m_obs,m_rewards,m_dones,m_states;

    def get_state(self):
        m_states = []
        for i in range(self.n_envs):
            states = self.s_games[i].get_state()
            for j in range(self.n_snakes):
                m_states.append(states[j])
        return np.array(m_states)

    def display(self,verbose=False,limit = -1):
        if limit == -1:
            limit = self.n_envs;
        for i in range(min(self.n_envs,limit)):
            print ('PLAYER : ',i)
            self.s_games[i].display(verbose=verbose)


    def close(self):
        pass

