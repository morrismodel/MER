
# MER
Nonlinear Monte Carlo Method for Imbalanced Data Learning

Nonlinear Monte Carlo Method is the application of nonlinear expectation (Peng, 2005). In machine learning, we substitute the mean value of loss function with the maximum value of subgroup mean loss, which follows the assumption of sublinear expectation.

****

Dev requirement:

```
tensorflow-gpu-1.13.1
cvxopt-1.2.5
```

For toydata:
```
python MER_for_toydata.py
```

For DNN+MER: 


```
python 10-nonlinearized-dual-multi.py 4 2 100
```
Here, args[1] = total subgroup number $N$;  \\
args[2] = number of minor-class subgroup $c$ \\
args[3] = number of data in each subgroup $s_j$ ;



For ResNet+MER:

```
python nonlinearized-dual-ResNet.py 4 2 100
```
