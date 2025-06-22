import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, Vector3
import torch
from gazebo_msgs.msg import ModelStates, EntityState
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

import tf_transformations

from std_msgs.msg import Float32

import os
# https://chatgpt.com/c/67bb9412-cd78-8009-9793-f75bc0af4192
# https://github.com/reiniscimurs/DRL-robot-navigation/blob/main/TD3/velodyne_env.py
# https://github.com/m5823779/motion-planner-reinforcement-learninxtg/blob/master/src/critic_network.py
# https://github.com/rcampbell95/turtlebot3_ddpg/blob/master/src/ddpg_stage_1.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from collections import deque

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0

    def add_value(self, value):
        if len(self.values) == self.window_size:
            self.sum -= self.values[0]
        self.values.append(value)
        self.sum += value

    def get_average(self):
        if len(self.values) == 0:
            return 0.0
        return self.sum / len(self.values)

# Example usage
# window_size = 5
# moving_average = MovingAverage(window_size)

# values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# for value in values:
#     moving_average.add_value(value)
#     print(f"Added value: {value}, Moving Average: {moving_average.get_average()}")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Use deque for efficient removal of oldest element when full
        self.capacity = capacity
        self.count = 0

    def push(self, state, action, reward, next_state, done):
        """ Save a transition """
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.count+=1
        a_array = action[0]*np.ones((1,1))
        if self.count < self.capacity:
            self.buffer.append((state, a_array, reward, next_state, done))
        else:
            self.buffer.popleft()
            self.buffer.append((state, a_array, reward, next_state, done))
        

    def sample(self, batch_size):
        """ Sample a batch of transitions from the buffer """
        if self.count < batch_size:
            transitions = random.sample(self.buffer, self.count)
        else:
            transitions = random.sample(self.buffer, batch_size)
            
        # Unzip the transitions into separate lists
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert lists to NumPy arrays with consistent shapes
        states = np.concatenate(states, axis=0)
        # print('actions', actions)
        try:
            actions = np.concatenate(actions, axis=0)
        except:
            # print('exception catching')
            # Ensure all actions have the same number of dimensions
            actions = [np.expand_dims(action, axis=0) if action.ndim == 1 else action for action in actions]
            actions = np.concatenate(actions, axis=0)  
        
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.concatenate(next_states, axis=0)
        dones = np.array(dones).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones        

    #     if self.count < batch_size:
    #         transitions = random.sample(self.buffer, self.count)
    #     else:
    #         transitions = random.sample(self.buffer, batch_size)

    #     batch = map(np.array, zip(*transitions))
    #     print('batch', batch)
    #     return batch

    # def __len__(self):
    #     return len(self.buffer)
    
    

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x



class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.modelstate_subscriber = self.create_subscription(ModelStates, '/model_states_demo', self.modelstate_callback, 10)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        # self.reset_timer = self.create_timer(5, self.reset_callback)
        
        self.reward_avg_publisher = self.create_publisher(Float32, 'reward_avg', 10)
        self.action_publisher = self.create_publisher(Float32, 'action', 10)
        self.dist2goal_publisher = self.create_publisher(Float32, 'dist2goal', 10)
        self.reward_publisher = self.create_publisher(Float32, 'reward', 10)
        self.qval_publisher = self.create_publisher(Float32, 'qval', 10)
        self.client_pause = self.create_client(Empty, '/pause_physics')
        self.client_resetworld = self.create_client(Empty, '/reset_world')
        self.client_setentity =  self.create_client(SetEntityState, '/set_entity_state')
        self.client_unpause = self.create_client(Empty, '/unpause_physics')
        
        self.state_dim = 2 #distance to goal, pitch, yaw, velocity 
        self.action_dim = 1
        self.max_action = 1.

        self.reward_avg_array = MovingAverage(25)

        
        self.load = False
        
        
        if self.load:
            self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = optim.Adam(self.actor.parameters())



            self.critic = Critic(self.state_dim, self.action_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = optim.Adam(self.critic.parameters())
            
            self.load_model(directory='/home/ian/models', filename='model')
        else:
            
                 
            self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = optim.Adam(self.actor.parameters())

            self.critic = Critic(self.state_dim, self.action_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = optim.Adam(self.critic.parameters())
            
        self.replay_buffer = ReplayBuffer(100000)  # Instantiate the replay buffer with a capacity of 10000        
        
        self.evaluating = False
        
        self.current_eval_episode = 0
        self.iteration = 0
        self.batch_size = 50
        self.discount = 0.999
        self.tau = 0.01
        self.policy_noise = 10.0
        self.noise_clip = 20.0
        self.max_episodes = 10000
        self.expl_min = 0.1
        self.expl_max = 1.0
        self.expl_decay_steps = 1000000
        self.exploration_rate = self.expl_max
        self.episode_timesteps = 0
        self.total_timesteps = 0
        self.episode = 1
        self.reward = 0
        
        self.action = np.array([0.])
        self.next_action = np.array([0.])   
        self.state = np.array([ 0., 0.]) 
        self.next_state = np.array([0., 0.])    
    
        self.eval_episode_freq = 100
        self.eval_episodes = 3
        
        self.max_episodes = 100000
        self.max_steps = 500
        self.goal_threshold = 3.9
        
        self.done = False
        
        self.goalx = 17.
        self.goaly = 0.
        self.goalz = 1.
        
        
        self.startx = 2.
        self.starty = 0.
        self.startz = 15.0

        self.currentx = 0.
        self.currenty = 0.
        self.currentz = 15.0
        
        self.dist2goal = np.sqrt( (self.goalx - self.currentx)**2 + (self.goaly - self.currenty)**2 )
        self.start2goal = np.sqrt( (self.goalx - self.startx)**2 + (self.goaly - self.starty)**2 )
        self.linear = 1.0

        self.angular = 0.
        self.velocity = 0.1
        
        self.yaw = 0.
        self.pitch = 0.
        
        
        self.twist_lin = 0.

    def evaluate(self, network, epoch, eval_episodes=5):
        avg_reward = 0.0
        col = 0
        for _ in range(eval_episodes):
            count = 0
            state = env.reset()
            done = False
            while not done and count < 501:
                action = network.get_action(np.array(state))
                a_in = [(action[0] + 1) / 2, action[1]]
                state, reward, done, _ = env.step(a_in)
                avg_reward += reward
                count += 1
                if reward < -90:
                    col += 1
        avg_reward /= eval_episodes
        avg_col = col / eval_episodes
        print("..............................................")
        print(
            "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
            % (eval_episodes, epoch, avg_reward, avg_col)
        )
        print("..............................................")
        return avg_reward

    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
    
        # Convert lists to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        states = states.reshape(self.batch_size, self.state_dim)
        actions = actions.reshape(self.batch_size, self.action_dim)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        # print('states', states)
        # print('actions', actions)
        # print(actions.shape)
        with torch.no_grad():

            next_actions = self.actor_target(next_states)
            next_actions = next_actions.reshape(self.batch_size, self.action_dim)

            # print('next_actions', next_actions)
            # print(next_actions.shape)

            next_states = next_states.reshape(self.batch_size, self.state_dim)
            # print('next_states', next_states)
            # print(next_states.shape)

            target_Q_values = self.critic_target(next_states, next_actions)
            target_Q_values = rewards + (1 - dones) * discount * target_Q_values
        
        qval_pub = Float32()
        
        # Update the Critic network
        current_Q_values = self.critic(states, actions)
        qval_pub.data = current_Q_values.mean().item()
        self.qval_publisher.publish(qval_pub)
        # print('current_Q_values', current_Q_values)
        # print('target_Q_values', target_Q_values)
        critic_loss = F.mse_loss(current_Q_values, target_Q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update the target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    def get_state(self):
        
        return np.array([self.dist2goal, self.velocity])    
    
        
    def modelstate_callback(self, msg):
        
        self.currentx = msg.pose[3].position.x
        self.currenty = msg.pose[3].position.y
        self.currentz = msg.pose[3].position.z        
        self.dist2goal = np.sqrt( (self.goalx - self.currentx)**2 + (self.goaly - self.currenty)**2 )

        self.velocity = msg.twist[3].linear.x
        self.angular = msg.twist[3].angular.z
        
        
        quaternion = msg.pose[3].orientation
        quaternion_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        r ,self.pitch, self.yaw = tf_transformations.euler_from_quaternion(quaternion_list)
     

    def get_action(self, state):
        # Function to get the action from the actor
        # print('state', state)
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def step(self, action):
        pass


    def get_reward(self):
        reward = (self.start2goal - self.dist2goal) / self.start2goal - 1.0*self.velocity
        return reward

    def timer_callback(self):
        # print('self.currentz', self.currentz)
        if self.episode % self.eval_episode_freq == 0:
            self.evaluating = True
            self.episode+=1
            print('starting evaluation!')
         
        if self.current_eval_episode == 5:
            self.current_eval_episode = 0
            self.evaluating = False   
            print('stopping evaluation')
        
        if self.evaluating:
            self.action = self.get_action(self.state)
            
            self.episode_timesteps += 1
            self.total_timesteps += 1
            
            self.reward += self.get_reward()
            self.state = self.next_state
            self.action = self.next_action
            self.next_state = self.get_state()
            self.next_action = self.get_action(self.next_state)
            # self.next_action = (self.next_action + np.random.normal(0, self.policy_noise, size=self.action_dim)).clip(-self.noise_clip, self.noise_clip)
            self.next_action = self.next_action.reshape(1, self.action_dim)
            # self.twist_lin += (self.next_action[0][0])
            self.twist_lin = 10. * self.next_action[0][0]

            
            
            msg = Twist()
            msg.linear.x = self.twist_lin  
            msg.angular.z = 0.    
            self.publisher_.publish(msg)

            if abs(self.currenty) > 2.0:
                self.done = True
                print('exceeded y dimension\r\n\r\n\r\n')
                self.reward -= 100
                self.episode_timesteps = 0

                
            if self.currentx < -4.0:
                self.done = True
                print('exceeded x dimension\r\n\r\n\r\n')
                self.reward -= 100
                self.episode_timesteps = 0
            
            if self.currentz < 1.0:
                self.done = True
                print('missed the landing pad\r\n\r\n\r\n')
                self.reward -= 100
                self.episode_timesteps = 0
                

            if self.currentx > self.goalx + 5.:
                self.done = True
                print('went past the goal\r\n\r\n\r\n')
                self.reward -= 50
                self.episode_timesteps = 0
        

            if self.episode_timesteps==self.max_steps:
                self.done = True
                print('exceeded max steps')
                self.reward -= 50
                self.episode_timesteps = 0


            if (self.dist2goal < self.goal_threshold) and self.currentz < 3. and self.currentz > 1.:
                self.done = True
                print('reached goal\r\n\r\n\r\n')
                self.reward += 500
                self.episode_timesteps = 0

            if self.done:

                self.reset_callback()
                self.done = False
                self.reward = 0
        
        if self.evaluating==False:  
            # print('currentz', self.currentz)
            # print('self.dist2goal', self.dist2goal)
            
            self.action = self.get_action(self.state)
            
            self.episode_timesteps += 1
            self.total_timesteps += 1
            
            self.reward += self.get_reward()
            self.state = self.next_state
            self.action = self.next_action
            self.next_state = self.get_state()
            self.next_action = self.get_action(self.next_state)
            self.next_action = (self.next_action + np.random.normal(0, self.policy_noise, size=self.action_dim)).clip(-self.noise_clip, self.noise_clip)
            self.next_action = self.next_action.reshape(1, self.action_dim)
            # self.twist_lin += (self.next_action[0][0])
            self.twist_lin = 10. * self.next_action[0][0]
            action_msg = Float32()
            action_msg.data = self.next_action[0][0]
            self.action_publisher.publish(action_msg)
            
            msg = Twist()
            msg.linear.x = self.twist_lin    
            msg.angular.z = 0.    
            self.publisher_.publish(msg)

            if abs(self.currenty) > 2.0:
                self.done = True
                print('exceeded y dimension\r\n\r\n\r\n\r\n')
                self.reward -= 250
                self.episode+=1
                self.episode_timesteps = 0

                
            if self.currentx < -4.0:
                self.done = True
                print('exceeded x dimension\r\n\r\n\r\n\r\n')
                self.reward -= 100
                self.episode+=1
                self.episode_timesteps = 0

            if self.currentx > self.goalx +5.:
                self.done = True
                print('went past the goal\r\n\r\n\r\n')
                self.reward -= 250
                self.episode+=1
                self.episode_timesteps = 0
        

            if self.episode_timesteps==self.max_steps:
                self.done = True
                print('exceeded max steps\r\n\r\n\r\n')
                self.reward -= 250
                self.episode+=1
                self.episode_timesteps = 0


            if (self.dist2goal < self.goal_threshold) and self.currentz < 3. and self.currentz > 1.:
                self.done = True
                print('reached goal\r\n\r\n\r\n')
                self.reward += 500
                self.episode+=1
                self.episode_timesteps = 0

            if self.currentz < 1.0:
                self.done = True
                print('missed the landing pad\r\n\r\n\r\n')
                self.reward -= 100
                self.episode+=1
                self.episode_timesteps = 0
                
                
            self.replay_buffer.push(self.state, self.action, self.reward, self.next_state, self.done)


            if self.done:
                distgoal_msg = Float32()
                distgoal_msg.data = self.dist2goal
                self.dist2goal_publisher.publish(distgoal_msg)
                self.reset_callback()
                self.done = False
                self.reward = 0

    def reset_callback(self):
        self.get_logger().info('RESETTING robot')

        self.twist_lin = 0.
        
        self.client_pause.call_async(Empty.Request())
        if self.episode % 5 == 0:
            self.save_model('/home/ian/models', 'model')
        if self.evaluating:

            print('finished current evaluation episode')
            print('reward: ', self.reward)
            self.current_eval_episode += 1
            print('current_eval_episode', self.current_eval_episode)
        if self.evaluating==False:
            self.reward_avg_array.add_value(self.reward)
            r_msg = Float32()
            r_msg.data = self.reward_avg_array.get_average()
            self.reward_avg_publisher.publish(r_msg)
            print('episode count: ', self.episode)
            print('episode timesteps: ', self.episode_timesteps)
            print('total timesteps: ', self.total_timesteps)
            print('reward: ', self.reward)
            r_msg = Float32()
            r_msg.data = self.reward
            self.reward_publisher.publish(r_msg)
            self.reward = 0
            print('training')
            self.train(self.replay_buffer, self.iteration, batch_size=self.batch_size)
        self.iteration += 1
        # self.client_resetworld.call_async(Empty.Request())
        self.client_setentity.call_async(SetEntityState.Request(
            state=EntityState(
                name='burger',
                pose=Pose(
                    position=Point(x=self.startx, y=self.starty, z=self.startz),
                    orientation=Quaternion(x=0., y=0., z=0., w=1.)
                ),
                twist=Twist(
                    linear=Vector3(x=0., y=0., z=0.),
                    angular=Vector3(x=0., y=0., z=0.)
                )
            )
        ))
        
        self.goalx = np.random.uniform(low = 0., high = 0.01) + 17.   
        
        self.client_setentity.call_async(SetEntityState.Request(
            state=EntityState(
                name='landing_pad',
                pose=Pose(
                    position=Point(x=self.goalx, y=self.goaly, z=0.1),
                    orientation=Quaternion(x=0., y=0., z=0., w=1.)
                ),
                twist=Twist(
                    linear=Vector3(x=0., y=0., z=0.),
                    angular=Vector3(x=0., y=0., z=0.)
                )
            )
        ))

        self.start2goal = np.sqrt( (self.goalx - self.startx)**2 + (self.goaly - self.starty)**2 ) 
        self.client_unpause.call_async(Empty.Request())

    def save_model(self, directory, filename):
        """
        Save the actor and critic models to the specified directory with the given filename.
        
        :param directory: The directory where the models should be saved.
        :param filename: The base filename for the saved models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        actor_path = os.path.join(directory, f"{filename}_actor.pth")
        critic_path = os.path.join(directory, f"{filename}_critic.pth")
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        print(f"Models saved to {directory} with base filename {filename}")

    def load_model(self, directory, filename):
        """
        Load the actor and critic models from the specified directory with the given filename.
        
        :param directory: The directory where the models are saved.
        :param filename: The base filename for the saved models.
        """
        actor_path = os.path.join(directory, f"{filename}_actor.pth")
        critic_path = os.path.join(directory, f"{filename}_critic.pth")
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)
            print(f"Models loaded from {directory} with base filename {filename}")
        else:
            print(f"Model files not found in {directory} with base filename {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()