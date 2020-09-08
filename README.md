<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# MER
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
args[1] refers $N$; args[2] refers $s_j$
```
python 10-nonlinearized-dual-multi.py 4 100
```

For ResNet+MER:

```
python nonlinearized-dual-ResNet.py 4 100
```
