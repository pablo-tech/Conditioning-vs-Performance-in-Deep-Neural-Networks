# Deep Neural Network Conditioning vs Performance

### Team  
- Jakub Dworakowski, Computer Science, Stanford University   
- Pablo Rodriguez Bertorello, Computer Science, Stanford University 

### Abstract
We investigate the effects of neural network regularization techniques. First, we reason formally through the effect of dropout and training stochasticity on gradient descent. Then, we conduct classification experiments on the ImageNet data set, as well as regression experiments in the OneNow Reinforcement Learning data set. A network layer's weight matrix is quantified via Singular Value Decomposition and Conditioning ratios. Our regression network appeared to be well conditioned.  However, we find that learning for large-scale classification applications is likely to be capped by poor conditioning. We propose approaches that may prove breakthroughs in learning, providing early evidence.  We introduce a gradient perturbation layer, a method to maximize generalization experimentally. Our numerical analysis showcases the opportunity to introduce network circuitry compression, relying on the principal components of a layer's weights, when conditioning peaks. Generally, we propose conditioning as an objective function constraint.

### Report
For details see the project report https://github.com/pablo-tech/DeepNetwork-Conditioning-vs-Performance/blob/master/Project_Report_FINAL.pdf

### Summary
The project studies neural network conditioning under regularization approaches including Stochastic Gradient Descent.
![picture](img/stochastic-vs-batch-gradient-descent.png)
