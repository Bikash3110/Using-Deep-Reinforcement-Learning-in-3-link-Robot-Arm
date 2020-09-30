import numpy as np
import pyglet

from random import seed
from random import randint

class ArmEnv(object):
    # Pyglet specific viewer, can use others like pygame
    viewer = None
    dt = .1    # refresh rate
    # corresponds to up and down 
    action_bound = [-1, 1]
    #Specify goal
    goal = {'x': 100., 'y': 100., 'l': 40}
    # State dimensions and Action dimensions
    state_dim =  13    
    action_dim = 3     #3 joints
    
    
    def __init__(self):
        
        # tracks arm lenght and radius
        self.arm_info = np.zeros(
            3, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100        # 3 arms length
        self.arm_info['r'] = np.pi/6    # 3 angles information
        # boolean which tracks whether arm end is at goal(1) or not (0)
        self.on_goal = 0
    #=============================================================================================    
    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        
        # a1xy origin of joint, a1xy_ point where 1st arm and 2nd arm link
        # finger is point where 2nd arm link ends and 3rd start, end is point where 3rd arm link ends. 
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2) and finger starts
        end = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + finger # finger ends
        
        # represent the state of arm w.r.t distance
        # normalize features 
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        dist3 = [(self.goal['x'] - end[0]) / 400, (self.goal['y'] - end[1]) / 400]
        r = -np.sqrt(dist3[0]**2+dist3[1]**2)

        # reward function
        if self.goal['x'] - self.goal['l']/2 < end[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < end[1] < self.goal['y'] + self.goal['l']/2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/200, finger/200, end/200, dist1 + dist2 + dist3, [1. if self.on_goal else 0.]))
        return s, r, done
    #==================================================================================================    
    # Reset 
    def reset(self):
        self.goal['x'] = np.random.rand()*400.
        self.goal['y'] = np.random.rand()*400.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(3)
        self.on_goal = 0
        
        
        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        end = np.array([np.cos(a1r + a2r+ a3r), np.sin(a1r + a2r + a3r)]) * a3l + finger
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        dist3 = [(self.goal['x'] - end[0])/400, (self.goal['y'] - end[1])/400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, end/200, dist1 + dist2 + dist3, [1. if self.on_goal else 0.]))

        return s
#======================================================================================================================

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        #return np.random.rand(2)-0.5    # two radians
        return np.random.rand(3)-0.5    # three radians
        #======================================================================
        
class Viewer(pyglet.window.Window):
    bar_thc = 5     # bar thickness 
    #Viewer Width and Height
    #wid = 400 
    #ht = 400

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, training can be speed up
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = np.array([200, 200])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        #==================ADD ARM 3-Link ==========================================
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

        self.arm3 = self.batch.add(
           4, pyglet.gl.GL_QUADS, None,
            ('v2f', [50, 50,              # location
                     50, 60,
                     140, 60,
                     140, 50]), ('c3B', (249, 86, 86) * 4,))
        #=================ARM 3 LINKS Added==========================================
        
        
    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)
        
        #=========================================================================================================
        # calulate joint position
        (a1l, a2l, a3l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        a3xy_ = np.array([np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)]) * a2l + a2xy_
        # Figure out joints rotation     
        #a1tr, a2tr , a3tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - (self.arm_info['r'][0]+self.arm_info['r'][1]), np.pi / 2 - self.arm_info['r'].sum()
        a1tr = np.pi / 2 - self.arm_info['r'][0]
        a2tr = np.pi / 2 - (self.arm_info['r'][0]+self.arm_info['r'][1])
        a3tr = np.pi / 2 - self.arm_info['r'].sum()
        # Rotate 4 vertices of each 3-link rectangles 
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
 
        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        
        xy21_ = a2xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy22_ = a2xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy31 = a3xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy32 = a3xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        
        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        self.arm3.vertices = np.concatenate((xy21_, xy22_, xy31, xy32))
        #=========================================================================================================


    # convert the mouse coordinate to goal's coordinate, Evaluation type 1
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y
    
    # Evaluation type 2: random spawn 
    def on_close(self):
        self.goal_info['x'] = self.rand(400) #width
        self.goal_info['y'] = self.rand(400) #Heigth
        
    
    # random coordinate generators     
    def rand(self, x):
        value = randint(0,x) 
        if value > (x - (self.goal_info['l']/2)):
            value = value - self.goal_info['l']/2
        if value < (self.goal_info['l']/2):
            value = value + self.goal_info['l']/2
        return value
 
if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())
