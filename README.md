
# MER
Nonlinear Monte Carlo Method for Imbalanced Data Learning

\rule[-10pt]{14.3cm}{0.05em}

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
python 10-nonlinearized-dual-multi.py 4 100
```
Here, args[1] refers total subgroup number $N$; 
args[2] refers $s_j$


For ResNet+MER:

```
python nonlinearized-dual-ResNet.py 4 100
```
