---
title: Reinforcement Learning Algorithms - Value-based methods - [Part 5]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2025-02-01 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Reinforcement Learning Algorithms - Value-based methods - [Part 5]

## Tabular Learning and the Bellman Equation

### Value, state, and optimality

We deﬁned this value as an expected total
reward (optionally discounted) that is obtainable from the state. In a
formal way, the value of the state is given by

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-1.png?raw=true)

where $r_t$ is the local reward obtained at step $t$ of the episode.

The total reward could be discounted with $0 < γ < 1$ or not
discounted (when $γ = 1$); it’s up to us how to deﬁne it. The value is
always calculated in terms of some policy that our agent follows.

To
illustrate this, let’s consider a very simple environment with three
states

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-2.png?raw=true)

1.The agent’s initial state.  

2.The final state that the agent is in after executing action “right” from the initial state.   The reward obtained from this is 1.  

3.The final state that the agent is in after action “down.”   The reward obtained from this is 2.

The   environment is   always deterministic — every   action succeeds and we always start from state 1.   Once we reach either state 2 or state 3, the episode ends.   Now, the question is, what’s the value of state 1?   This question is meaningless without information about our agent’s behavior or, in other words, its policy.   Even in a simple environment, our agent can have an infinite amount of behaviors, each of which will have its own value for state 1.   Consider these examples:

- Agent always goes right  

- Agent always goes down  

- Agent goes right with a probability of 50% and down with a probability of 50%  

- Agent goes right in 10% of cases and in 90% of cases executes the “down” action

To demonstrate how the value is calculated, let’s do it for all the preceding policies:

- The value of state 1 in the case of the “always right” agent is    1.0    (every time it goes left, it obtains 1 and the episode ends)  

- For the “always down” agent, the value of state 1 is    2.0   

- For the 50% right/50% down agent, the value is 1   .   0   x   0   .   5+2   .   0   x   0   .   5 =    1    .    5   

- For the 10% right/90% down agent, the value is 1   .   0   x   0   .   1+2   .   0   x   0   .   9 =    1    .    9

Now, another question: what’s the optimal policy for this agent?   The goal of RL is to get as much total reward as possible.   For this one-step environment, the total reward is equal to the value of state 1, which, obviously, is at the maximum at policy 2 (always down).  

Unfortunately, such simple environments with an obvious optimal policy are not that interesting in practice.   For interesting environments, the optimal policies are much harder to formulate and it’s even harder to prove their optimality.   However, don’t worry; we are moving toward the point when we will be able to make computers learn the optimal behavior on their own.  

From the preceding example, you may have a false impression that we should always take the action with the highest reward.   In general, it’s not that simple.   To demonstrate this, let’s extend our preceding environment   with yet another state that   is reachable   from state 3.   State 3 is no longer a terminal state but a transition to state 4, with a bad reward of -20.   Once we have chosen the “down” action in state 1, this bad reward is unavoidable, as from state 3, we have only one exit to state 4.   So, it’s a trap for the agent, which has decided that “being greedy” is a good strategy.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-3.png?raw=true)

With that addition, our values for state 1 will be calculated this way:

- The “always right” agent is the same: 1.0

- The “always down” agent gets 2 . 0 + ( − 20) = − 18

- The 50%/50% agent gets 0 . 5 x 1 . 0 + 0 . 5 x (2 . 0 + ( − 20)) = − 8 . 5

- The 10%/90% agent gets 0 . 1 x 1 . 0 + 0 . 9 x (2 . 0 + ( − 20)) = − 16.1

So, the best policy for this new environment is now policy 1: always
go right. We spent some time discussing naïve and trivial
environments so that you realize the complexity of this optimality
problem and can appreciate the results of Richard Bellman be er.
Bellman was an American mathematician who formulated and
proved his famous Bellman equation. We will talk about it in the
next section.

### The Bellman equation of optimality

Let’s start with a deterministic case, when all our actions
have a 100% guaranteed outcome. Imagine that our agent observes
state $s_0$ and has $$ available actions. Every action leads to another
state, $s_1 … s_N$ , with a respective reward, $r_1 … r_N$ . Also, assume
that we know the values, $V_i$ , of all states connected to state $s_0$ .
What will be the best course of action that the agent can take in such
a state?

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-4.png?raw=true)

If we choose the concrete action,   $a_i$    , and calculate the value given to this action, then the value will be   $V_0 (   a   =   a_i    ) =   r_i    +   V_i$    .   So, to choose the best possible action, the agent needs to calculate the resulting values for every action and choose the maximum possible outcome.   In other words,   $V_0    = max_{a   ∈   1   …   N}    (   r_a    +   V_a    )$.   If we are using the discount factor,   $γ$   , we need to multiply the value of the next state by gamma:   $V_0    = max_{a   ∈   1   …   N}    (   r_a    +   γV_a    )$.

This may look very similar to our greedy example from the previous section, and, in fact, it is.   However, there is one difference: when we act greedily, we do not only look at the immediate reward for the action, but at the immediate reward plus the long-term value of the state.   This allows us to avoid a possible trap with a large immediate reward but a state that has a bad value.  

Bellman proved that with that extension, our behavior will get the best possible outcome.   In other words, it will be optimal.   So, the preceding equation is called the Bellman equation of value (for a deterministic case).  

It’s not very complicated to extend this idea for a stochastic case, when our actions have the chance of ending up in different states.   What we need to do is calculate the expected value for every action, instead of just taking the value of the next state.

To illustrate this, let’s consider one single action available from state   $s_0$    , with three possible outcomes:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-5.png?raw=true)


Here, we have   one   action, which can lead to three different states with different probabilities.   With probability   $p_1$    , the action can end up in state   $s_1$    , with   $p_2$    in state   $s_2$    , and with   $p_3$    in state   $s_3    (   p_1    +   p_2    +   p_3    = 1)$.   Every target state has its own reward $(   r_1    ,   r_2    , or   r_3    )$.   To calculate the expected value after issuing action 1, we need to sum all values, multiplied by their probabilities:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-6.png?raw=true)

or, more formally

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-7.png?raw=true)


Here,   $𝔼_{s   ∼   S}$    means taking the expected value over all states in our state space,   S   .

By combining the Bellman equation, for a deterministic case, with a value for stochastic actions, we get the Bellman optimality equation for a general case:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-8.png?raw=true)

Note that   $p_{a,i   →   j}$    means the probability of action   a   , issued in state   i   , ending up in state   j   .   The interpretation is still the same: the optimal value of the state corresponds to the action, which gives us the maximum possible expected immediate reward, plus the discounted long-term reward for the next state.   You may also notice that this definition is recursive: the value of the state is defined via the values of the immediately reachable states.   This recursion may look like cheating: we define some value, pretending that we already know it.   However, this is a very powerful and common technique in computer science and even in math in general (proof by induction is based on the same trick).   This Bellman equation is a foundation not only in RL but also in much more general dynamic programming, which is a widely used method for solving practical optimization problems.  

These values not only give us the best reward that we can obtain, but they basically give us the optimal policy to obtain that reward: if our agent knows the value for every state, then it automatically knows how to gather this reward.   Thanks to Bellman’s optimality proof, at every state the agent ends up in, it needs to select the action with the maximum expected reward, which is a sum of the immediate reward and the one-step discounted long-term reward – that’s it.   So, those values are really useful to know.   Before you get familiar with a practical way   to calculate   them, I need to introduce one more mathematical notation.   It’s not as fundamental as the value of the state, but we need it for our convenience.

### The value of the action

To make our   life slightly easier, we can define different quantities, in addition to the value of the state,   $V   (   s   )$, as the value of the action,   $Q   (   s,a   )$.   Basically, this equals the total reward we can get by executing action   a   in state   s   and can be defined via   $V   (   s   )$.   Being a much less fundamental entity than   $V   (   s   )$, this quantity gave a name to the whole family of methods called Q-learning, because it is more convenient.   In these methods, our primary objective is to get values of   Q   for every pair of state and action:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-9.png?raw=true)

Q   , for this state,   s   , and action,   a   , equals the expected immediate reward and the discounted long-term reward of the destination state.   We also can define   $V   (   s   )$ via   $Q   (   s,a   )$:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-10.png?raw=true)

This just means that the value of some state equals to the value of the maximum action we can execute from this state.

Finally, we can express   $Q   (   s,a   )$ recursively:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-11.png?raw=true)

In the last formula, the index on the immediate reward, $(   s,a   )$, depends on the environment details:

- If the immediate reward is given to us after executing a particular action,   a   , from state   s   , index (   s,a   ) is used and the formula is exactly as shown above.  

- But if the reward is provided for reaching some state,   s'   , via action   a'   , the reward will have the index (   s'   ,a'   ) and will need to be moved into the max operator:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-12.png?raw=true)

That difference   is not very significant from a mathematical point of view, but it could be important during the implementation of the methods.   The first situation is more common, so we will stick to the preceding formula.

To give you a concrete example, let’s consider an environment that is similar to FrozenLake, but has a much simpler structure: we have one initial state $(   s_0    )$ surrounded by four target states,   $s_1$    ,   $s_2$    ,   $s_3$    ,   $s_4$    , with different rewards:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-13.png?raw=true)

Every action is probabilistic in the same way as in FrozenLake: with a 33% chance that our action will be executed without modifications, but with a 33% chance that we will slip to the left, relatively, of our target cell and a 33% chance that we will slip to the right.   For simplicity, we use discount factor   $γ   = 1$.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-14.png?raw=true)

Let’s calculate the values of the actions to begin with.   Terminal states   $s_1    …   s_4$    have no outbound connections, so   Q   for those states is   zero for all actions.   Due to this, the values of the terminal states are equal to their immediate reward (once we get there, our episode ends without any subsequent states):   $V_1    = 1$,   $V_2    = 2$,   $V_3    = 3$,   $V_4    = 4$.

The values of the actions for state 0 are a bit more complicated.   Let’s start with the “up” action.   Its value, according to the definition, is equal to the expected sum of the immediate reward plus the long-term value for subsequent steps.   We have no subsequent steps for any possible transition for the “up” action:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-15.png?raw=true)

Repeating this for the rest of the   $s_0$    actions results in the following:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-16.png?raw=true)

The final value for state   s    0    is the maximum of those actions’ values, which is 2.97.  
Q-values are much more convenient in practice, as for the agent, it’s much simpler to make decisions about actions based on   Q   than on   V   .   In the case of   Q   , to choose the action based on the state, the agent just needs to calculate   Q   for all available actions using the current state and choose the action with the largest value of   Q   .   To do the same using values of the states, the agent needs to know not only the values, but also the probabilities for transitions.   In practice, we   rarely know them in advance, so the agent needs to estimate transition probabilities for every action and state pair.   Later in this chapter, you will see this in practice by solving the FrozenLake environment both ways.   However, to be able to do this, we have one important thing still missing: a general way to calculate   $V_i$    and   $Q_i$    .

### The value iteration method

In the simplistic   example you just saw, to calculate the values of the states and actions, we exploited the structure of the environment: we had no loops in transitions, so we could start from terminal states, calculate their values, and then proceed to the central state.   However, just one loop in the environment builds an obstacle in our approach.   Let’s consider such an environment with two states:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-17.png?raw=true)


We start from state   $s_1$    , and the only action we can take leads us to state   $s_2$    .   We get the reward,   $r   = 1$, and the only transition from   $s_2$    is an action, which brings us back to   $s_1$    .   So, the life of our agent is an infinite sequence of states $[   s_1    ,s_2    ,s_1    ,s_2    ,   …   ]$.   To deal with this infinity loop, we can use a discount factor:   $γ   = 0   .   9$.   Now, the question is, what are the values for both the states?   The answer is not very complicated, in fact.   Every transition from   $s_1$    to   $s_2$    gives us a reward of 1 and every back transition gives us 2.   So, our sequence of rewards will be $[1   ,   2   ,   1   ,   2   ,   1   ,   2   ,   1   ,   2   ,   …   ]$.   As there is only one action available in every state, our agent has no choice, so we can omit the    max    operation in formulas (there is only one alternative).

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-18.png?raw=true)

Strictly speaking, we can’t calculate the exact values for our states, but with   $γ   = 0   .   9$, the contribution of every transition quickly decreases over time.   For example, after 10 steps,   $γ_{10}    = 0   .   9    10    ≈   0   .   349$, but after 100 steps, it becomes just $0   .   0000266$.   Due to this, we can stop after 50 iterations and still get quite a precise estimation:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-19.png?raw=true)

The preceding example can be used to get the gist of a more general
procedure called the value iteration algorithm . This allows us to
numerically calculate the values of the states and values of the
actions of Markov decision processes ( MDPs ) with known
transition probabilities and rewards. The procedure (for values of
the states) includes the following steps:

1.Initialize the values of all states,   $V_i$    , to some initial value (usually zero) 

2.For every state,   s   , in the MDP, perform the Bellman update:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-20.png?raw=true)

3.Repeat step 2 for some   large number of steps or until changes become too small

Okay, so that’s the theory.   In practice, this method has certain obvious limitations.   First of all, our state space should be discrete and small enough to perform multiple iterations over all states.   This is not an issue for FrozenLake-4x4 and even for FrozenLake-8x8 (it exists in Gym as a more challenging version), but for CartPole, it’s not totally clear what to do.   Our observation for CartPole is four float values, which represent some physical characteristics of the system.   Potentially, even a small difference in those values could have an influence on the state’s value.   One of the solutions for that could be discretization of our observation’s values; for example, we can split the observation space of CartPole into bins and treat every bin as an individual discrete state in space.   However, this will create lots of practical problems, such as how large bin intervals should be and how much data from the environment we will need to estimate our values.

The second practical problem arises from the fact that we rarely know the transition probability for the actions and rewards matrix.   Remember the interface provided by Gym to the agent’s writer: we observe the state, decide on an action, and only then do we get the next observation and reward for the transition.   We don’t know (without peeking into Gym’s environment code) what the probability is of getting into state   $s_1$    from state   $s_0$    by issuing action   $a_0$    .   What we do have is just the history from the agent’s interaction with the environment.   However, in Bellman’s update, we need both a reward for every transition and the probability of this transition.   So, the obvious answer to this issue is to use our agent’s experience as an estimation for both unknowns.   Rewards could be used as they are.   We just need to remember what reward we got on the transition from   $s_0$    to   $s_1$    using action   $a$   , but to estimate probabilities, we   need to maintain counters for every tuple $(   s_0    ,s_1    ,a   )$ and normalize them.

### Value iteration in practice

In this section, we will look at how the value iteration method will
work for FrozenLake.

```python
#!/usr/bin/env python3
import typing as tt
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter

ENV_NAME = "FrozenLake-v1"
#ENV_NAME = "FrozenLake8x8-v1"      # uncomment for larger version
GAMMA = 0.9
TEST_EPISODES = 20

State = int
Action = int
RewardKey = tt.Tuple[State, Action, State]
TransitKey = tt.Tuple[State, Action]


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.rewards: tt.Dict[RewardKey, float] = defaultdict(float)
        self.transits: tt.Dict[TransitKey, Counter] = defaultdict(Counter)
        self.values: tt.Dict[State, float] = defaultdict(float)

    def play_n_random_steps(self, n: int):
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, is_done, is_trunc, _ = self.env.step(action)
            rw_key = (self.state, action, new_state)
            self.rewards[rw_key] = float(reward)
            tr_key = (self.state, action)
            self.transits[tr_key][new_state] += 1
            if is_done or is_trunc:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def calc_action_value(self, state: State, action: Action) -> float:
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            rw_key = (state, action, tgt_state)
            reward = self.rewards[rw_key]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def select_action(self, state: State) -> Action:
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, is_trunc, _ = env.step(action)
            rw_key = (state, action, new_state)
            self.rewards[rw_key] = float(reward)
            tr_key = (state, action)
            self.transits[tr_key][new_state] += 1
            total_reward += reward
            if is_done or is_trunc:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print(f"{iter_no}: Best reward updated {best_reward:.3} -> {reward:.3}")
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
```

The central data structures in this
example are as follows:

- Reward table : A dictionary with the composite key “source
state” + “action” + “target state.” The value is obtained from the
immediate reward.

- Transitions table : A dictionary keeping counters of the
experienced transitions. The key is the composite “state” +
“action,” and the value is another dictionary that maps the
“target state” into a count of times that we have seen it.
For example, if in state 0 we execute action 1 ten times, after
three times, it will lead us to state 4 and after seven times to
state 5. Then entry with the key (0, 1) in this table will be a dict
with the contents { 4: 3, 5: 7 } . We can use this table to estimate
the probabilities of our transitions.

- Value table : A dictionary that maps a state into the calculated
value of this state.

The overall logic of our code is simple: in the loop, we play 100
random steps from the environment, populating the reward and
transition tables. After those 100 steps, we perform a value iteration
loop over all states, updating our value table. Then we play several
full episodes to check our improvements using the updated value
table. If the average reward for those test episodes is above the 0.8
boundary, then we stop training. During the test episodes, we also
update our reward and transition tables to use all data from the
environment.

The function calc_action_value() calculates the value of the action from the state using our transition, reward, and values tables.   We will use it for two purposes: to select the best action to perform from the state and to calculate the new value of the state on value iteration.

We do the following:  

1.We extract transition counters for the given state and action from the transition table.   Counters in this table have a form of    dict    , with target states as the key and a count of experienced transitions as the value.   We sum all counters to obtain the total count of times we have executed the action from the state.   We will use this total value later to go from an individual counter to probability.  

2.Then we   iterate every target state that our action has landed on and calculate its contribution to the total action value using the Bellman equation.   This contribution is equal to immediate reward plus discounted value for the target state.   We multiply this sum to the probability of this transition and add the result to the final action value.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-21.png?raw=true)

In the preceding diagram, we do a calculation of the value for state s
and action a . Imagine that, during our experience, we have executed
this action several times $( c_1 + c_2 )$ and it ends up in one of two
states, $s_1$ or $s_2$ . How many times we have switched to each of these
states is stored in our transition table as dict ${ s_1 : c_1 , s_2 : c_2 }$ .

Then, the approximate value for the state and action, $Q ( s,a )$, will be
equal to the probability of every state, multiplied by the value of the
state. From the Bellman equation, this equals the sum of the
immediate reward and the discounted long-term state value.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-22.png?raw=true)

Our solution is stochastic, and my experiments usually required 10
to 100 iterations to reach a solution, but in all cases, it took less than
a second to ﬁnd a good policy that could solve the environment in
80% of runs. If you remember, about an hour was needed to achieve
a 60% success ratio using the cross-entropy method, so this is a
major improvement. There are two reasons for that.

First, the stochastic outcome of our actions, plus the length of the
episodes (6 to 10 steps on average), makes it hard for the cross-
entropy method to understand what was done right in the episode
and which step was a mistake. Value iteration works with individual
values of the state (or action) and incorporates the probabilistic
outcome of actions naturally by estimating probability and
calculating the expected value. So, it’s much simpler for value
iteration and requires much less data from the environment (which
is called sample eﬃciency in RL).

The second reason is the fact that value iteration doesn’t need full
episodes to start learning. In an extreme case, we can start updating
our values just from a single example. However, for FrozenLake, due
to the reward structure (we get 1 only after successfully reaching the
target state), we still need to have at least one successful episode to
start learning from a useful value table, which may be challenging to
achieve in more complex environments. For example, you can try
switching the existing code to a larger version of FrozenLake, which
has the name FrozenLake8x8-v1 . The larger version of FrozenLake
can take from 150 to 1,000 iterations to solve, and, according to
TensorBoard charts, most of the time it waits for the ﬁrst successful
episode, then it very quickly reaches convergence.

### Q-iteration for pratice 

The most obvious change is to our value table. In the previous
example, we kept the value of the state, so the key in the
dictionary was just a state. Now we need to store values of the
Q-function, which has two parameters, state and action , so thekey in the value table is now a composite of ( State , Action )
values.

The second diﬀerence is in our calc _action _value() function.
We just don’t need it anymore, as our action values are stored in
the value table.

Finally, the most important change in the code is in the agent’s
value _iteration() method. Before, it was just a wrapper around
the calc _action _value() call, which did the job of Bellman
approximation. Now, as this function has gone and been
replaced by a value table, we need to do this approximation in
the value _iteration() method.

```python
#!/usr/bin/env python3
import typing as tt
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter

ENV_NAME = "FrozenLake-v1"
#ENV_NAME = "FrozenLake8x8-v1"      # uncomment for larger version
GAMMA = 0.9
TEST_EPISODES = 20

State = int
Action = int
RewardKey = tt.Tuple[State, Action, State]
TransitKey = tt.Tuple[State, Action]


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.rewards: tt.Dict[RewardKey, float] = defaultdict(float)
        self.transits: tt.Dict[TransitKey, Counter] = \
            defaultdict(Counter)
        self.values: tt.Dict[TransitKey, float] = defaultdict(float)

    def play_n_random_steps(self, n: int):
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, is_done, is_trunc, _ = \
                self.env.step(action)
            rw_key = (self.state, action, new_state)
            self.rewards[rw_key] = float(reward)
            tr_key = (self.state, action)
            self.transits[tr_key][new_state] += 1
            if is_done or is_trunc:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def select_action(self, state: State) -> Action:
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, is_trunc, _ = \
                env.step(action)
            rw_key = (state, action, new_state)
            self.rewards[rw_key] = float(reward)
            tr_key = (state, action)
            self.transits[tr_key][new_state] += 1
            total_reward += reward
            if is_done or is_trunc:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    rw_key = (state, action, tgt_state)
                    reward = self.rewards[rw_key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print(f"{iter_no}: Best reward updated "
                  f"{best_reward:.3} -> {reward:.3}")
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
```

## Deep Q-Networks

### Real-life value iteration

The improvements that we got in the FrozenLake environment by
switching from the cross-entropy method to the value iteration
method are quite encouraging, so it’s tempting to apply the value
iteration method to more challenging problems. However, it is
important to look at the assumptions and limitations that our value
iteration method has. But let’s start with a quick recap of the
method. On every step, the value iteration method does a loop on
all states, and for every state, it performs an update of its value with
a Bellman approximation. The variation of the same method for Q-
values (values for actions) is almost the same, but we approximate
and store values for every state and action. So what’s wrong with
this process?

The ﬁrst obvious problem is the count of environment states and
our ability to iterate over them. In value iteration, we assume that
we know all states in our environment in advance, can iterate over
them, and can store their value approximations. It’s easy to do for
the simple grid world environment of FrozenLake, but what about
other tasks?

To understand this, let’s ﬁrst look at how scalable the value
iteration approach is, or, in other words, how many states we can
easily iterate over in every loop. Even a moderate-sized computer
can keep several billion ﬂoat values in memory (8.5 billion in 32 GB
of RAM), so the memory required for value tables doesn’t look like a huge constraint. Iteration over billions of states and actions will
be more central processing unit ( CPU )-demanding but is not an
insurmountable problem.

Nowadays, we have multicore systems that are mostly idle, so by
using parallelism, we can iterate over billions of values in a
reasonable amount of time. The real problem is the number of
samples required to get good approximations for state transition
dynamics. Imagine that you have some environment with, say, a
billion states (which corresponds approximately to a FrozenLake of
size 31600 × 31600). To calculate even a rough approximation for
every state of this environment, we would need hundreds of
billions of transitions evenly distributed over our states, which is
not practical.

To give you an example of an environment with an even larger
number of potential states, let’s consider the Atari 2600 game
console again. This was very popular in the 1980s, and many arcade-
style games were available for it. The Atari console is archaic by
today’s gaming standards, but its games provide an excellent set of
RL problems that humans can master fairly quickly, yet are still
challenging for computers. Not surprisingly, this platform (using
an emulator, of course) is a very popular benchmark within RL
research, as I mentioned.

Let’s calculate the state space for the Atari platform. The resolution
of the screen is 210 × 160 pixels, and every pixel has one of 128
colors. So every frame of the screen has 210 ⋅ 160 = 33600 pixels and
the total number of diﬀerent screens possible is $128^33600$ , which is
slightly more than $10^70802$ . If we decide to just enumerate all possible states of the Atari once, it will take billions of billions of
years even for the fastest supercomputer. Also, 99(.9)% of this job
will be a waste of time, as most of the combinations will never be
shown during even long gameplay, so we will never have samples
of those states. However, the value iteration method wants to iterate
over them just in case.

The second main problem with the value iteration approach is that
it limits us to discrete action spaces. Indeed, both $Q ( s,a )$ and $V (s)$ approximations assume that our actions are a mutually exclusive
discrete set, which is not true for continuous control problems
where actions can represent continuous variables, such as the angle
of a steering wheel, the force on an actuator, or the temperature of a
heater. This issue is much more challenging than the ﬁrst, and we
will talk about it in the last part of the book, in chapters dedicated
to continuous action space problems. For now, let’s assume that we
have a discrete count of actions and that this count is not very large
(i.e., orders of 10s). How should we handle the state space size
issue?

### Tabular Q-learning

The key question to focus on when trying to handle the state space
issue is, do we really need to iterate over every state in the state
space? We have an environment that can be used as a source of
real-life samples of states. If some state in the state space is not
shown to us by the environment, why should we care about its
value? We can only use states obtained from the environment to
update the values of states, which can save us a lot of work.

This modiﬁcation of the value iteration method is known as Q-
learning, as mentioned earlier, and for cases with explicit state-to-
value mappings, it entails the following steps:

1. Start with an empty table, mapping states to values of actions.

2. By interacting with the environment, obtain the tuple s , a , r , s
′ (state, action, reward, and the new state). In this step, you
need to decide which action to take, and there is no single
proper way to make this decision.

3. Update the $Q ( s,a )$ value using the Bellman approximation:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-23.png?raw=true)

As in value iteration, the end condition could be some threshold of the update, or we could perform test episodes to estimate the expected reward from the policy.   Another thing to note here is how to update the Q-values.   As we take samples from the environment, it’s generally a bad idea to just assign new values on top of existing values, as training can become unstable.

What is usually done in practice is updating the   Q   (   s,a   ) with approximations using a “blending” technique, which is just averaging between old and new values of Q using learning rate   α   with a value from 0 to 1:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-24.png?raw=true)

This allows values of Q to converge smoothly, even if our environment is noisy.   The final version of the algorithm is as follows:

1.Start with an empty table for   $Q   (   s,a   )$.  

2.Obtain $(   s   ,   a   ,   r   ,   s'   )$ from the environment.

3.Make a Bellman update:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-25.png?raw=true)

4.Check convergence conditions.   If not met, repeat from step 2.

As mentioned earlier, this method is called tabular Q-learning, as we keep a table of states   with their Q-values.   Let’s try it on our FrozenLake environment.

```python
#!/usr/bin/env python3
import typing as tt
import gymnasium as gym
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter

ENV_NAME = "FrozenLake-v1"
#ENV_NAME = "FrozenLake8x8-v1"      # uncomment for larger version
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

State = int
Action = int
ValuesKey = tt.Tuple[State, Action]

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.values: tt.Dict[ValuesKey] = defaultdict(float)

    def sample_env(self) -> tt.Tuple[State, Action, float, State]:
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, is_tr, _ = self.env.step(action)
        if is_done or is_tr:
            self.state, _ = self.env.reset()
        else:
            self.state = new_state
        return old_state, action, float(reward), new_state

    def best_value_and_action(self, state: State) -> tt.Tuple[float, Action]:
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, state: State, action: Action, reward: float, next_state: State):
        best_val, _ = self.best_value_and_action(next_state)
        new_val = reward + GAMMA * best_val
        old_val = self.values[(state, action)]
        key = (state, action)
        self.values[key] = old_val * (1-ALPHA) + new_val * ALPHA

    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, is_tr, _ = env.step(action)
            total_reward += reward
            if is_done or is_tr:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        state, action, reward, next_state = agent.sample_env()
        agent.value_update(state, action, reward, next_state)

        test_reward = 0.0
        for _ in range(TEST_EPISODES):
            test_reward += agent.play_episode(test_env)
        test_reward /= TEST_EPISODES
        writer.add_scalar("reward", test_reward, iter_no)
        if test_reward > best_reward:
            print("%d: Best test reward updated %.3f -> %.3f" % (iter_no, best_reward, test_reward))
            best_reward = test_reward
        if test_reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
```

### Deep Q-learning

The Q-learning method that we have just covered solves the issue
of iteration over the full set of states, but it can still struggle with
situations when the count of the observable set of states is very
large. For example, Atari games can have a large variety of diﬀerent
screens, so if we decide to use raw pixels as individual states, we
will quickly realize that we have too many states to track and
approximate values for.

In some environments, the count of diﬀerent observable states
could be almost inﬁnite. For example, in CartPole, the environment
gives us a state that is four ﬂoating point numbers. The number of
value combinations is ﬁnite (they’re represented as bits), but this
number is extremely large. With just bit values, it is around $2^{4 ⋅ 32} ≈
3 . 4 ⋅ 10^{38}$ . In reality, it is less, as state values of the environment
are bounded, so not all bit combinations of 4 ﬂoat32 values are
possible, but the resulting state space is still too large. We could
create some bins to discretize those values, but this often creates
more problems than it solves; we would need to decide what ranges
of parameters are important to distinguish as diﬀerent states and
what ranges could be clustered together. As we’re trying to
implement RL methods in a general way (without looking inside
the environment’s internals), this is not a very promising direction.

In the case of Atari, one single pixel change doesn’t make much
diﬀerence, so we might want to treat similar images as one state.
However, we still need to distinguish some of the states.

As a solution to this problem, we can use a nonlinear
representation that maps both the state and action onto a value. In
machine learning, this is called a “regression problem.” The
concrete way to represent and train such a representation can vary,
but, as you may have already guessed from this section’s title, using
a deep NN is one of the most popular options, especially when
dealing with observations represented as screen images. With this
in mind, let’s make modiﬁcations to the Q-learning algorithm:

1. Initialize Q ( s,a ) with some initial approximation.

2. By interacting with the environment, obtain the tuple ( s , a , r ,s' ).

3. Calculate the loss:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-26.png?raw=true)

4. Update $Q ( s,a )$ using the stochastic gradient descent (SGD)
algorithm, by minimizing the loss with respect to the model
parameters.

5. Repeat from step 2 until converged.

This algorithm looks simple, but, unfortunately, it won’t work very
well. Let’s discuss some of the aspects that could go wrong and the
potential ways we could approach these scenarios.

### Interaction with the environment

First of all, we need to interact with the environment somehow to
receive data to train on. In simple environments, such as FrozenLake, we can act randomly, but is this the best strategy to
use? Imagine the game of Pong. What’s the probability of winning
a single point by randomly moving the paddle? It’s not zero, but
it’s extremely small, which just means that we will need to wait for
a very long time for such a rare situation. As an alternative, we can
use our Q-function approximation as a source of behavior (as we
did before in the value iteration method, when we remembered our
experience during testing).

If our representation of Q is good, then the experience that we get
from the environment will show the agent relevant data to train on.
However, we’re in trouble when our approximation is not perfect
(at the beginning of the training, for example). In such a case, our
agent can be stuck with bad actions for some states without ever
trying to behave diﬀerently. This is the exploration versus
exploitation.

On the one hand, our agent needs to explore
the environment to build a complete picture of transitions and
action outcomes. On the other hand, we should use interaction with
the environment eﬃciently; we shouldn’t waste time by randomly
trying actions that we have already tried and learned outcomes for.

As you can see, random behavior is be er at the beginning of the
training when our Q approximation is bad, as it gives us more
uniformly distributed information about the environment states.
As our training progresses, random behavior becomes ineﬃcient,
and we want to fall back to our Q approximation to decide how to
act.

A method that performs such a mix of two extreme behaviors is
known as an epsilon-greedy method , which just means switching
between random and Q policy using the probability
hyperparameter $𝜖$ . By varying $𝜖$ , we can select the ratio of random
actions. The usual practice is to start with $𝜖 = 1 . 0$ (100% random
actions) and slowly decrease it to some small value, such as 5% or
2% random actions. Using an epsilon-greedy method helps us to
both explore the environment in the beginning and stick to good
policy at the end of the training. There are other solutions to the
exploration versus exploitation problem. This problem is one of the
fundamental open questions in RL and an active area of research
that is not even close to being resolved completely.

### SGD optimization

The core of our Q-learning procedure is borrowed from supervised
learning. Indeed, we are trying to approximate a complex, nonlinear
function, $Q ( s,a )$, with an NN. To do this, we must calculate targets
for this function using the Bellman equation and then pretend that
we have a supervised learning problem at hand. That’s okay, but
one of the fundamental requirements for SGD optimization is that
the training data is independent and identically distributed
(frequently abbreviated as iid ), which means that our training data
is randomly sampled from the underlying dataset we’re trying to
learn on.

In our case, data that we are going to use for the SGD update
doesn’t fulﬁll these criteria:

1. Our samples are not independent. Even if we accumulate a
large batch of data samples, they will all be very close to each
other, as they will belong to the same episode.

2. Distribution of our training data won’t be identical to samples
provided by the optimal policy that we want to learn. Data that
we have will be a result of some other policy (our current
policy, a random one, or both in the case of epsilon-greedy),
but we don’t want to learn how to play randomly: we want an
optimal policy with the best reward.

To deal with this nuisance, we usually need to use a large buﬀer of
our past experience and sample training data from it, instead of
using our latest experience. This technique is called a replay buﬀer .
The simplest implementation is a buﬀer of a ﬁxed size, with new
data added to the end of the buﬀer so that it pushes the oldest
experience out of it.

The replay buﬀer allows us to train on more-or-less independent
data, but the data will still be fresh enough to train on samples
generated by our recent policy.

### Correlation between steps

Another practical issue with the default training procedure is also
related to the lack of iid data, but in a slightly diﬀerent manner. The
Bellman equation provides us with the value of $Q ( s,a )$ via $Q ( s' ,a' )$ (this process is called bootstrapping , when we use the formula
recursively). However, both the states s and s ′ have only one step between them. This makes them very similar, and it’s very hard for
NNs to distinguish between them. When we perform an update of
our NNs’ parameters to make $Q ( s,a )$ closer to the desired result,
we can indirectly alter the value produced for $Q ( s' ,a' )$ and other
states nearby. This can make our training very unstable, like
chasing our own tail; when we update Q for state s , then on
subsequent states, we will discover that $Q ( s' ,a' )$ becomes worse
but a empts to update it can spoil our $Q ( s,a )$ approximation even
more, and so on.

To make training more stable, there is a trick, called target network
, by which we keep a copy of our network and use it for the $Q ( s' ,a' )$ value in the Bellman equation. This network is synchronized with
our main network only periodically, for example, once in N steps
(where N is usually quite a large hyperparameter, such as 1k or 10k
training iterations).

### The Markov property

Our RL methods use Markov decision process ( MDP ) formalism
as their basis, which assumes that the environment obeys the
Markov property: observations from the environment are all that we
need to act optimally. In other words, our observations allow us to
distinguish states from one another.

One
single image from the Atari game is not enough to capture all the
important information (using only one image, we have no idea
about the speed and direction of objects, like the ball and our
opponent’s paddle). This obviously violates the Markov property and moves our single-frame Pong environment into the area of
partially observable MDPs ( POMDPs ). A POMDP is basically an
MDP without the Markov property, and it is very important in
practice. For example, for most card games in which you don’t see
your opponents’ cards, game observations are POMDPs because
the current observation (i.e., your cards and the cards on the table)
could correspond to diﬀerent cards in your opponents’ hands. 

The solution is maintaining several observations from the
past and using them as a state. In the case of Atari games, we
usually stack k subsequent frames together and use them as the
observation at every state. This allows our agent to deduct the
dynamics of the current state, for instance, to get the speed of the
ball and its direction. The usual “classical” number of k for Atari is
four. Of course, it’s just a hack, as there can be longer dependencies
in the environment, but for most of the games, it works well.

### The final form of DQN training

[Ref](https://www.nature.com/articles/nature14236)

The algorithm for DQN from the preceding papers has the
following steps: 

1.Initialize parameters for   Q   (   s,a   ) and Q̂(   s,a   ) with random weights,   𝜖   ←   1   .   0, and empty the replay buffer. 

2.With probability   𝜖   , select a random action   a   ; otherwise,   a   = arg max    a    Q   (   s,a   ).  

3.Execute action   a   in an emulator and observe the reward,   r   , and the next state,   s   ′   .  

4.Store the transition (   s   ,   a   ,   r   ,   s   ′   ) in the replay buffer.  

5.Sample a random mini-batch of transitions from the replay buffer.  

6.For every transition in the buffer, calculate the target:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-27.png?raw=true)

7.Calculate the loss:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-5-28.png?raw=true)

8.Update   Q   (   s,a   ) using the SGD algorithm   by minimizing the loss in respect to the model parameters.  

9.Every   N   steps, copy weights from   Q   to Q̂.  

10.Repeat from step 2 until converged.

### Code

Tackling Atari   games with RL is quite demanding from a resource perspective.   To make things faster, several transformations are applied to the Atari platform interaction, which are described in DeepMind’s paper.   Some of these transformations influence only performance, but some address Atari platform features that make learning long and unstable.   Transformations are implemented as Gym wrappers of various kinds.   The full list is quite lengthy and there are several implementations of the same wrappers in various sources.

#### Train

wrapper.py

```python
import typing as tt
import gymnasium as gym
from gymnasium import spaces
import collections
import numpy as np
from stable_baselines3.common import atari_wrappers


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(),
            shape=new_shape, dtype=obs.dtype)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0), obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype)
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(self, *, seed: tt.Optional[int] = None, options: tt.Optional[dict[str, tt.Any]] = None):
        for _ in range(self.buffer.maxlen-1):
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset()
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)


def make_env(env_name: str, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = atari_wrappers.AtariWrapper(env, clip_reward=False, noop_max=0)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env
```

dqn_model.py

```python
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.ByteTensor):
        # scale on GPU
        xx = x / 255.0
        return self.fc(self.conv(xx))
```

dqn_pong.py

```python
#!/usr/bin/env python3
import gymnasium as gym
import dqn_model
import wrappers

from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard.writer import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,           # current state
    torch.LongTensor,           # actions
    torch.Tensor,               # rewards
    torch.BoolTensor,           # done || trunc
    torch.ByteTensor            # next state
]

@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tt.List[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net: dqn_model.DQN, device: torch.device,
                  epsilon: float = 0.0) -> tt.Optional[float]:
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, is_tr, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(
            state=self.state, action=action, reward=float(reward),
            done_trunc=is_done or is_tr, new_state=new_state
        )
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done or is_tr:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return states_t.to(device), actions_t.to(device), rewards_t.to(device), \
           dones_t.to(device),  new_states_t.to(device)


def calc_loss(batch: tt.List[Experience], net: dqn_model.DQN, tgt_net: dqn_model.DQN,
              device: torch.device) -> torch.Tensor:
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_t
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, device, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
                  f"eps {epsilon:.2f}, speed {speed:.2f} f/s")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()
    writer.close()
```

#### Play

```python
#!/usr/bin/env python3
import gymnasium as gym
import argparse
import numpy as np
import typing as tt

import torch

import wrappers
import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", required=True, help="Directory for video")
    args = parser.parse_args()

    env = wrappers.make_env(args.env, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c: tt.Dict[int, int] = collections.Counter()

    while True:
        state_v = torch.tensor(np.expand_dims(state, 0))
        q_vals = net(state_v).data.numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] += 1
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        if is_done or is_trunc:
            break
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()

```




