import numpy as np
import tensorflow as tf

from memory import Memory

class DQNAgent(object):
    
    
    def __init__(self,
                 state_shape,
                 n_actions,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon=1,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 target_update_iter=500,
                 mem_size=50_000,
                 batch_size=32,
                 dueling=True
    ):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_iter = target_update_iter
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.dueling = dueling
        
        
        self.steps = 0
        self.mem = Memory(state_shape, mem_size)
        
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess.run(self.target_replace_op)
        
        tf.summary.FileWriter("logs/", self.sess.graph)


    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.state_shape], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.state_shape], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.not_done = tf.placeholder(tf.float32, [None, ], name='a')
        
        w_initializer = tf.random_normal_initializer()
        b_initializer = tf.random_normal_initializer()
        
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 64, tf.nn.relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name="e1")
            
            if self.dueling:
                self.V_eval = tf.layers.dense(e1, 1, tf.nn.relu,kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name="v_eval")
                self.A_eval = tf.layers.dense(e1, self.n_actions, tf.nn.relu,kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name="a_eval")
                self.q_eval = self.V_eval + (self.A_eval - tf.reduce_mean(self.A_eval, axis=1, keep_dims=True))
            else:
                self.q_eval = tf.layers.dense(e1, self.n_actions, tf.nn.relu,kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name="q")
                
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="t1")
            if self.dueling:
                self.V_target = tf.layers.dense(t1, 1, tf.nn.relu,kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name="v_target")
                self.A_target = tf.layers.dense(e1, self.n_actions, tf.nn.relu,kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name="a_target")
                self.q_next = self.V_target + (self.A_target - tf.reduce_mean(self.A_target, axis=1, keep_dims=True))
            else:
                self.q_next = tf.layers.dense(t1, self.n_actions, tf.nn.relu,kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name="t2")
            
        with tf.variable_scope('q_target'):
            a_eval= tf.argmax(self.q_eval, axis=1, output_type=tf.int32)
            a_indices_eval = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), a_eval], axis=1)
            q_next_target = tf.gather_nd(params=self.q_next, indices=a_indices_eval)
            q_target = self.r + self.gamma * self.not_done * q_next_target
            self.q_target = tf.stop_gradient(q_target)
        
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            
    
    def is_mem_ready(self):
        return self.mem.is_ready
    
    def act(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            q_values = self.sess.run(self.q_eval, feed_dict={self.s: state})
            action = np.argmax(q_values)
        return action
    
    def store_experience(self, exp):
        self.mem.add(exp)
    
    def learn(self):
        if self.steps % self.target_update_iter == 0:
            self.sess.run(self.target_replace_op)
        
        idx = self.mem.sample(self.batch_size)
        s = self.mem.data[idx][:, :self.state_shape]
        s_ = self.mem.data[idx][:, -self.state_shape:]
        a = self.mem.data[idx][:, self.state_shape].astype(int)
        r = self.mem.data[idx][:, self.state_shape + 1]
        not_done = 1 - self.mem.data[idx][:, self.state_shape + 2]

        _, _ = self.sess.run(
                [self._train_op, self.loss],
                feed_dict={
                        self.s: s,
                        self.s_: s_,
                        self.a: a,
                        self.r: r,
                        self.not_done: not_done,
                        }
                )
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        
        self.steps +=1
            
