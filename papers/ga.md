# ES, GA

The difference between ES and GA is not consistent between papers.


https://openai.com/blog/evolution-strategies/
https://eng.uber.com/deep-neuroevolution/


From OpenAI:
- Evolution Strategies as a Scalable Alternative to Reinforcement learning
    They applied a traditional ES with self-mutation adn elite without crossover.

From Uber:


- Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning
    First paper in the series of 5. They also tested NS, the technique known in the classic ES, and this GA-NS outperforms GA in some envs.
- Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients
    : Let NN determine an adaptive mutation rate
- [optional] On the Relationship Between the OpenAI Evolution Strategy and Stochastic Gradient Descent
- [optional] ES Is More Than Just a Traditional Finite Difference Approximator
- Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents
    Based on previously presented proof-of-concept level GA-NS, it presents NS-ES, NSR-ES, NSRA-ES.
