## Feed-Forward Nueral Networks

### Topology

> when i say im using "chatgpt", what exactly does that mean? its the trained model right? do the frameworks have a name? is that what tenserflow is?

> are NN's state machines? if so, is the NN interpolating between unknown inputs? does that mean a NN doesnt work when inputs are outside the training range?

> is 'running' a NN just computer all the activation functions? is this why GPUs are so good for NN because all these linear operations can be easily parrallelised?

> are 'real' NN this simple?

> what makes up the topology of cutting edge networks?

- made up of layers
    - input layer
    - hidden layer(s)
    - output layer

    ![alt text](./images/ffnn.png)

- data flows one way (as opposed to recurrent networks which allow bidirectional flow)
- nodes perform non-linear functions on their inputs and output a real number (**Activation Function**)
- signal strength is controlled by a weight at each connection

### Hyperparameters

### Activation Functions

> when computers quantize the signals, do we run the risk of not satisfying the Universal Approximation Theorem (which proves a two-layer NN is a universal function approximator when used with non-linear activation functions)?

> are biases required?

- function performed by a node upon receiving signal(s)

### Training, Backpropogation

> when training finds a local minima, is it like the input produced the correct output, but did it differently than the other correct answers, so it kinda rewarded the right answer without looking at how it got to a solution like if you reward someone for passing a test but they cheated so they keep cheating?

> if i train a NN with garbage input like assigning a number to a letter randomly, how do we make sense of what the NN learned?

> are there multiple cost functions?

> could you perform gradient descent on a binary activation function by counting the changes in 0 and 1?



## Recurrent Neural Network

### Topology

> Minsky and Papert pointed out that recurrent networks can be unrolled in time into a layered feedforward network.[23]: 354 (https://en.wikipedia.org/wiki/Recurrent_neural_network)

- contains loops allowing memory
- good at sequential tasks where previous inputs affect the output


```python
rnn = RNN()
ff = FFN()
hidden_state = [0.0, 0.0, 0.0, ..., 0.0]

for item in input:
    output, hidden_state = rnn(item, hidden_state)

prediction = ff(output)
```
- short term memory (vanishing gradient)
    - long short-term memory (LSTM)
        - uses gates to remember relevant information and forget others
    - gated recurrent unit (GRU)
        - update gate
        - reset gate
        - less tensor operations than LSTM

### Training

- backpropogation
- short term memory