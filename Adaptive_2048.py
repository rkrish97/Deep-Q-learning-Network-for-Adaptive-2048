import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,Flatten
from keras.optimizers import Adam,RMSprop
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive 
from google.colab import auth
from oauth2client.client import GoogleCredentials
from collections import deque
import sys
from random import randint
from random import sample as sam
from collections import Counter

import six
class Engine:

  def __init__(self, N=4, start_tiles=2, seed=None):
    self.N = N
    self.score = 0
    self.ended = False
    self.won = False
    self.last_move = '-'
    self.start_tiles = start_tiles
    self.board = [[0]*self.N for i in range(self.N)]
    self.merged = [[False]*self.N for i in range(self.N)]
    self.action_space = [0,1,2,3]
    if seed:
        random.seed(seed)
    
    self.add_start_tiles()

  def sample(self):  ###### TEAM CODE
    a = np.random.randint(4)
    return(a)

  def reset_game(self):
    self.score = 0
    self.ended = False
    self.won = False
    self.last_move = '-'
    self.board = [[0]*self.N for i in range(self.N)]
    self.merged = [[False]*self.N for i in range(self.N)]
    
    self.add_start_tiles()
    f = []
    for sublist in self.board:
        for item in sublist:
            f.append(item)
    s = np.reshape(np.array(f), (1,1,4,4))
    return s 


  def get_board(self):
    return self.board


  def add_start_tiles(self):
    for i in range(self.start_tiles):
      i1 = randint(0,3)
      j1 = randint(0,3)
      self.board[i1][j1] = 2 if random.random() < 0.9 else 4



  def add_random(self):   
    empty_cells_n = []
    empty_cells = []
    filled_cells = []
    fitted_rows = []
    fitted_columns = []
    for i in range(0, self.N):
        for j in range(0, self.N):
            if self.board[i][j] == 0:
                empty_cells += [[[i,j]]]
                if i ==0:
                    if j ==0:
                        if self.board[0][1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[0][1])
                        if self.board[1][0]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[1][0])
                    elif j ==self.N-1:
                        if self.board[0][self.N-2]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[0][self.N-2])
                        if self.board[1][self.N-1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[1][self.N-1])
                    else:
                        if self.board[0][j-1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[0][j-1])
                        if self.board[1][j]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[1][j])
                        if self.board[0][j+1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[0][j+1])
                elif i == self.N-1:
                    if j == 0:
                        if self.board[self.N-2][0]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[self.N-2][0])
                        if self.board[self.N-1][1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[self.N-1][1])
                    elif j == self.N-1:
                        if self.board[self.N-2][self.N-1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[self.N-2][self.N-1])
                        if self.board[self.N-1][self.N-2]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[self.N-1][self.N-2])
                    else:
                        if self.board[i][j-1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i][j-1])
                        if self.board[i][j+1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i][j+1])
                        if self.board[i-1][j]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i-1][j])
                else:
                    if j == 0:
                        if self.board[i-1][j]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i-1][j])
                        if self.board[i][j+1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i][j+1])
                        if self.board[i+1][j]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i+1][j])
                    elif j==self.N-1:
                        if self.board[i-1][j]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i-1][j])
                        if self.board[i][j-1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i][j-1])
                        if self.board[i+1][j]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i+1][j])
                    else:
                        if self.board[i-1][j]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i-1][j])
                        if self.board[i][j-1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i][j-1])
                        if self.board[i][j+1]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i][j+1])
                        if self.board[i+1][j]!=0:
                            empty_cells_n += [[i,j]]
                            filled_cells.append(self.board[i+1][j])
    empty_cells = np.array(empty_cells)
    empty_cells_n = np.array(empty_cells_n)
    rows = np.array(empty_cells[:,:,0])
    rows = rows[:,0]
    columns = np.array(empty_cells[:,:,1])
    columns = columns[:,0]
    rows = np.array([item for item, count in Counter(rows).items() if count == 1])
    columns = np.array([item for item, count in Counter(columns).items() if count == 1])
    
    flag=0;
    if rows.any() and columns.any():
        flag = randint(1,2)
    elif rows.any():
        flag = 1
    elif columns.any():
        flag=2
    else:
        flag=0
    if flag==1:
        for i in range(len(empty_cells_n)):
            if rows[0]==empty_cells_n[i][0]:
                cell = empty_cells_n[i]
                
                self.board[cell[0]][cell[1]] = 2 if random.random() < 0.9 else 4
                break
    elif flag==2:
        for i in range(len(empty_cells_n)):
            if columns[0]==empty_cells_n[i][1]:
                cell = empty_cells_n[i]
                
                self.board[cell[0]][cell[1]] = 2 if random.random() < 0.9 else 4
                break
    if empty_cells_n.any() and flag==0:
        
        m_val = 0
        index = 0
        for i in range(0,len(filled_cells)):
            if filled_cells[i] > m_val:
                index,m_val = i,filled_cells[i]
        ind = [index]
        for i in range(0,len(filled_cells)):
            if m_val == filled_cells[i]:
                ind.append(i)
        max_value = []
        for k in ind:
            if k not in max_value:
                max_value.append(k)
        
        cell = []
        
        for j in max_value:
            cell.append(empty_cells_n[j])
        
        n = []
        for r in cell:
            m = []
            if r[0]>=1:
                
                m.append(self.board[r[0]-1][r[1]])
            else:
                
                m.append(-1)
            if r[1]>=1:
                
                m.append(self.board[r[0]][r[1]-1])
            else:
                
                m.append(-1)
            if r[0]<3:
                
                m.append(self.board[r[0]+1][r[1]])
            else:
                
                m.append(-1)
            if r[1]<3:
                
                m.append(self.board[r[0]][r[1]+1])
            else:
                
                m.append(-1)
            n = np.append(n,m,axis = 0)
        n = np.reshape(n,(len(cell),4))
        n = np.asmatrix(n)
        n = np.array(n)
        count_two = []
        count_four = []
        for i in range(0,len(n)):
            twos = 0
            fours = 0
            for j in range(0,len(n[0])):
                if n[i][j] == 2:
                    twos = twos +1

                if n[i][j] == 4:
                    fours = fours +1

            count_two.append(twos)
            count_four.append(fours)
        if random.random() < 0.9:
            max_fo = np.argmax(np.array(count_four)- np.array(count_two))
            if max_fo.any():
                self.board[cell[max_fo][0]][cell[max_fo][1]] = 2
            else:
                ran = randint(0,len(cell)-1);
                self.board[cell[ran][0]][cell[ran][1]] = 2;


        else:
            max_two = np.argmax(np.array(count_two) - np.array(count_four))
            if max_two.any():
                self.board[cell[max_two][0]][cell[max_two][1]] = 4;
            else:
                ran = randint(0,len(cell)-1);
                self.board[cell[ran][0]][cell[ran][1]] = 4;

  def create_traversal(self, vector):
    v_x = list(range(0,self.N))
    v_y = list(range(0,self.N))

    if vector['x'] == 1:
      v_x.reverse() 
    elif vector['y'] == 1:
      v_y.reverse()
        
    return (v_y, v_x)


  def find_furthest(self, row, col, vector):
    found = False
    val = self.board[row][col]
    i = row + vector['y']
    j = col + vector['x']
    while i >= 0 and i < self.N and j >= 0 and j < self.N:
      val_tmp = self.board[i][j] 
      if self.merged[i][j] or (val_tmp != 0 and val_tmp != val):
          return (i - vector['y'], j - vector['x'])
      if val_tmp:
          return (i, j)
          
      i += vector['y']
      j += vector['x']
        
    return (i - vector['y'], j - vector['x'])


  def create_vector(self, direction):
    if direction == 0:
      return {'x': 0, 'y': -1}
    elif direction == 1:
      return {'x': 1, 'y': 0}
    elif direction == 2:
      return {'x': 0, 'y': 1}
    else:
      return {'x': -1, 'y': 0}


  def moves_available(self):
    moves = [False]*4
    for direction in range(4):
      dir_vector = self.create_vector(direction)
      traversal_y, traversal_x = self.create_traversal(dir_vector)        

      for row in traversal_y:
        for col in traversal_x:
          val = self.board[row][col]

          if val:
            n_row, n_col = self.find_furthest(row, col, dir_vector)

            if not ((n_row,n_col) == (row,col)):
              n_val = self.board[n_row][n_col]
              if (val == n_val and not self.merged[n_row][n_col]) or (n_val == 0):
                moves[direction] = True

    return moves


  def move(self, direction):
    # up: 0, right: 1, down: 2, left: 3
    dir_vector = self.create_vector(direction)
    traversal_y, traversal_x = self.create_traversal(dir_vector)        
    self.last_move = str(direction)
    reward = 0

    moved = False
    for row in traversal_y:
      h = 0
      g = 0
      for col in traversal_x:
        val = self.board[row][col]

        if val:
          g = g + 1
          n_row, n_col = self.find_furthest(row, col, dir_vector)

          # if furthest is found
          if not ((n_row, n_col) == (row,col)):
            # merge
            if val == self.board[n_row][n_col] and not self.merged[n_row][n_col]:
                self.board[n_row][n_col] += val
                self.board[row][col] = 0
                self.merged[n_row][n_col] = True

                reward += val*2
                self.score += reward
                self.won = (reward == 2048)
                moved = True
            # move
            elif self.board[n_row][n_col] == 0:
                self.board[n_row][n_col] += val
                self.board[row][col] = 0
                #reward = 0
                moved = True
          #else:
            #h = h + 1
      #if h == g:
        #reward = -1

    # reset merged flags
    self.merged = [[False]*self.N for i in range(self.N)]
    if moved:
        self.add_random()

    self.ended = not True in self.moves_available() or self.won 
    if self.ended and not self.won:
        reward = -1

    return reward, self.ended

class DQN:

  def __init__(self, env):   
    env = Engine()
    self.env     = env
    self.memory  = deque(maxlen=2000)
    self.learning_rate = 0.005
    self.tau = 0.1
    self.gamma = 0.9
    self.epsilon = 1
    self.epsilon_min = 0.05
    self.epsilon_decay = 0.99
    

    self.model        = self.create_model()
    self.target_model = self.create_model()

  

  def create_model(self):   
    model   = Sequential()
    model.add(Conv2D(32,(2,2), padding = 'valid', activation = "relu", input_shape = (1,4,4), data_format = "channels_first"))
    model.add(Conv2D(64,(2,2), padding = 'valid', activation = "relu"))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dense(len(self.env.action_space)))
    model.compile(loss="mean_squared_error",          
        optimizer=RMSprop(lr=self.learning_rate))          
    return model

  def act(self, state):              
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon_min, self.epsilon)
    if np.random.random() < self.epsilon:
        return self.env.sample()
    return np.argmax(self.model.predict(state))

  def remember(self, state, action, reward, new_state, done):
    self.memory.append([state, action, reward, new_state, done])

  def replay(self):    
    batch_size = 50
    if len(self.memory) < batch_size: 
        return
    samples = random.sample(self.memory, batch_size)
    for sample in samples:
        state, action, reward, new_state, done = sample
        target = self.target_model.predict(state)
        if not done:
          target[0][action] = reward +  (max(self.target_model.predict(new_state)[0])) * self.gamma    #optimal Q Value estimate
        else:
          target[0][action] = reward
            
        self.model.fit(state, target, epochs=1, verbose=0)

  def target_train(self):  
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
    self.target_model.set_weights(target_weights)

  def save_model(self, fn1, fn2):    
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    self.model.save(fn1)
    model_file = drive.CreateFile({'title' : fn1})
    model_file.SetContentFile(fn1)
    model_file.Upload()
    drive.CreateFile({'id': model_file.get('id')})
    self.target_model.save(fn2)
    model_file2 = drive.CreateFile({'title' : fn2})
    model_file2.SetContentFile(fn2)
    model_file2.Upload()
    drive.CreateFile({'id': model_file2.get('id')})
        

def main():   
  env = Engine()
  trials  = 10001      #number of episodes
  dqn_agent = DQN(env=env)
  
  for trial in range(trials):
    print(trial)
    cur_state = env.reset_game()
    move_number = 0
    dqn_agent.epsilon = 1
    if len(dqn_agent.memory)>1999:
      dqn_agent.memory.clear()
    
    while True:
      
      move_number = move_number+1
      
      action = dqn_agent.act(cur_state)    #picking the action
      reward, done = env.move(action)       #obtaining the reward
      f = []
      for sublist in env.board:
          for item in sublist:
              f.append(item)
      new_state = np.reshape(np.array(f), (1,1,4,4))   #reshaping the state of the environment to fit the input specs of the neural network

      dqn_agent.remember(cur_state, action, reward, new_state, done)   #memory buffer
      
      dqn_agent.replay() #training policy network
      dqn_agent.target_train()    #load weights from policy to target network

      cur_state = new_state
      if done:
        break
    print(move_number)
    
    if trial % 100 == 0:
      
      dqn_agent.save_model("mod-{}.h5".format(trial),"tar-{}.h5".format(trial))     #saving the policy and target model for every 100 episodes

if __name__ == "__main__":
    main()

def __test__(modelv):    
  e = Engine()
  e.__init__()
  count_256 = 0
  count_128 = 0
  score = []
  moves = []
  maxi2 = []
  episodes = 10000          #change this to modify the number of episodes
  for i in range(episodes):
    move  = 0
    e.reset_game()
    f=[]
    for sublist in e.board:
      for item in sublist:
          f.append(item)
    cur_state = np.reshape(np.array(f),(1,1,4,4))
    new_state = cur_state
    ended = False
    maxi = 0
    while not ended:
      move = move + 1
      pred = modelv.predict(new_state)
      if np.random.random() < 0.05:
        action = np.random.randint(4)
      else:
        action = np.argmax(pred)
      reward,ended=e.move(action)
      
      f = []
      for sublist in e.board:
          for item in sublist:
              f.append(item)
      new_state = np.reshape(np.array(f), (1,1,4,4))
    maxi = np.max(e.board)
    maxi2.append(maxi)
    if np.max(e.board)==256:
      count_256 = count_256+1
    elif np.max(e.board)==128:
      count_128 = count_128+1
    print(i)
    score.append(e.score)
    moves.append(move)
  print("Mean score : ",np.mean(score))
  print("Mean number of moves :",np.mean(moves))
  print("Max tile:",max(maxi2))
