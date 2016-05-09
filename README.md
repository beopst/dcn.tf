# Dynamic Capacity Networks using Tensorflow

Dynamic Capacity Networks (DCN; http://arxiv.org/abs/1511.07838) implementation using Tensorflow. DCN reduces the number of computations by applying high-capacity networks to only some selected input patches. 

## Dataset

* [Cluttered MNIST](https://github.com/deepmind/mnist-cluttered)

## Results

| Models | Validation accuracy |
|:------:|:-------------------:|
| Coarse Model | 2.492 |
| Fine Model | 1.070 |
| DCN without hint (4 patches) | 1.875 |
| DCN without hint (8 patches) | 1.438 |
| DCN without hint (16 patches) | 1.197 |
| DCN without hint (24 patches) | 1.069 |
| DCN with hint (4 patches) | 1.509 |
| DCN with hint (8 patches) | 1.197 |
| DCN with hint (16 patches) | 1.007 |
| DCN with hint (24 patches) | 0.933 |

Validation accuracy scores are averaged over the last 10 epochs.

## Notes

* It is observed that hint objective gives better generalization. Also, validation accuracy improves as the number of patches increases. With 24 patches, DCN with hint objective outperforms Fine Model. Note that the original paper uses 8 patches.

* In this benchmark, weight decay parameter was set to 0.0005, and additional weight parameter for hint objective was set to 0.01. For training, Adam optimizer was used. Initial learning rate was 0.001, and it was decreased a factor of 10 for every 100 epochs. Total number of epochs was 300. 

* Computation time issue
  
  In this implementation, computation time for DCN is slightly lower than that of Fine Model (in case of 8 patches).
   

## Logs
* Coarse Model
```
Step 226800 (epoch 290.03), Elapsed: 2358.3 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 226900 (epoch 290.15), Elapsed: 2342.8 ms, LR: 0.0000, Loss: 0.0079 Hint loss: 0.0000
Step 227000 (epoch 290.28), Elapsed: 2352.9 ms, LR: 0.0000, Loss: 0.0083 Hint loss: 0.0000
Step 227100 (epoch 290.41), Elapsed: 2345.2 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 227200 (epoch 290.54), Elapsed: 2343.2 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 227300 (epoch 290.66), Elapsed: 2357.4 ms, LR: 0.0000, Loss: 0.0148 Hint loss: 0.0000
Step 227400 (epoch 290.79), Elapsed: 2361.9 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 227500 (epoch 290.92), Elapsed: 2346.6 ms, LR: 0.0000, Loss: 0.0093 Hint loss: 0.0000

[Validation] Epoch: 291.00, Elapsed: 696.8 ms, Error: 2.45

Step 227600 (epoch 291.05), Elapsed: 2341.3 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 227700 (epoch 291.18), Elapsed: 2341.7 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 227800 (epoch 291.30), Elapsed: 2357.9 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 227900 (epoch 291.43), Elapsed: 2338.8 ms, LR: 0.0000, Loss: 0.0078 Hint loss: 0.0000
Step 228000 (epoch 291.56), Elapsed: 2349.2 ms, LR: 0.0000, Loss: 0.0079 Hint loss: 0.0000
Step 228100 (epoch 291.69), Elapsed: 2366.5 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 228200 (epoch 291.82), Elapsed: 2367.5 ms, LR: 0.0000, Loss: 0.0079 Hint loss: 0.0000
Step 228300 (epoch 291.94), Elapsed: 2431.5 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000

[Validation] Epoch: 292.00, Elapsed: 694.0 ms, Error: 2.53

Step 228400 (epoch 292.07), Elapsed: 2356.8 ms, LR: 0.0000, Loss: 0.0079 Hint loss: 0.0000
Step 228500 (epoch 292.20), Elapsed: 2349.3 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 228600 (epoch 292.33), Elapsed: 2358.9 ms, LR: 0.0000, Loss: 0.0078 Hint loss: 0.0000
Step 228700 (epoch 292.46), Elapsed: 2335.6 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 228800 (epoch 292.58), Elapsed: 2358.8 ms, LR: 0.0000, Loss: 0.0085 Hint loss: 0.0000
Step 228900 (epoch 292.71), Elapsed: 2360.5 ms, LR: 0.0000, Loss: 0.0085 Hint loss: 0.0000
Step 229000 (epoch 292.84), Elapsed: 2359.6 ms, LR: 0.0000, Loss: 0.0088 Hint loss: 0.0000
Step 229100 (epoch 292.97), Elapsed: 2337.0 ms, LR: 0.0000, Loss: 0.0083 Hint loss: 0.0000

[Validation] Epoch: 293.00, Elapsed: 688.8 ms, Error: 2.52

Step 229200 (epoch 293.09), Elapsed: 2351.8 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000
Step 229300 (epoch 293.22), Elapsed: 2360.4 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000
Step 229400 (epoch 293.35), Elapsed: 2345.0 ms, LR: 0.0000, Loss: 0.0078 Hint loss: 0.0000
Step 229500 (epoch 293.48), Elapsed: 2343.5 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 229600 (epoch 293.61), Elapsed: 2359.6 ms, LR: 0.0000, Loss: 0.0082 Hint loss: 0.0000
Step 229700 (epoch 293.73), Elapsed: 2367.9 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000
Step 229800 (epoch 293.86), Elapsed: 2362.6 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 229900 (epoch 293.99), Elapsed: 2351.2 ms, LR: 0.0000, Loss: 0.0084 Hint loss: 0.0000

[Validation] Epoch: 294.00, Elapsed: 703.0 ms, Error: 2.45

Step 230000 (epoch 294.12), Elapsed: 2355.6 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 230100 (epoch 294.25), Elapsed: 2349.7 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 230200 (epoch 294.37), Elapsed: 2355.4 ms, LR: 0.0000, Loss: 0.0087 Hint loss: 0.0000
Step 230300 (epoch 294.50), Elapsed: 2349.3 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 230400 (epoch 294.63), Elapsed: 2347.4 ms, LR: 0.0000, Loss: 0.0078 Hint loss: 0.0000
Step 230500 (epoch 294.76), Elapsed: 2344.4 ms, LR: 0.0000, Loss: 0.0087 Hint loss: 0.0000
Step 230600 (epoch 294.88), Elapsed: 2348.7 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000

[Validation] Epoch: 295.00, Elapsed: 689.6 ms, Error: 2.49

Step 230700 (epoch 295.01), Elapsed: 2351.1 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 230800 (epoch 295.14), Elapsed: 2350.2 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000
Step 230900 (epoch 295.27), Elapsed: 2340.4 ms, LR: 0.0000, Loss: 0.0089 Hint loss: 0.0000
Step 231000 (epoch 295.40), Elapsed: 2353.8 ms, LR: 0.0000, Loss: 0.0079 Hint loss: 0.0000
Step 231100 (epoch 295.52), Elapsed: 2357.5 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000
Step 231200 (epoch 295.65), Elapsed: 2371.2 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 231300 (epoch 295.78), Elapsed: 2423.6 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 231400 (epoch 295.91), Elapsed: 2348.8 ms, LR: 0.0000, Loss: 0.0079 Hint loss: 0.0000

[Validation] Epoch: 296.00, Elapsed: 699.8 ms, Error: 2.47

Step 231500 (epoch 296.04), Elapsed: 2337.9 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 231600 (epoch 296.16), Elapsed: 2362.0 ms, LR: 0.0000, Loss: 0.0079 Hint loss: 0.0000
Step 231700 (epoch 296.29), Elapsed: 2349.2 ms, LR: 0.0000, Loss: 0.0096 Hint loss: 0.0000
Step 231800 (epoch 296.42), Elapsed: 2357.2 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 231900 (epoch 296.55), Elapsed: 2353.0 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 232000 (epoch 296.68), Elapsed: 2358.1 ms, LR: 0.0000, Loss: 0.0085 Hint loss: 0.0000
Step 232100 (epoch 296.80), Elapsed: 2349.0 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000
Step 232200 (epoch 296.93), Elapsed: 2347.6 ms, LR: 0.0000, Loss: 0.0075 Hint loss: 0.0000

[Validation] Epoch: 297.00, Elapsed: 701.5 ms, Error: 2.54

Step 232300 (epoch 297.06), Elapsed: 2351.3 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 232400 (epoch 297.19), Elapsed: 2357.8 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 232500 (epoch 297.31), Elapsed: 2336.1 ms, LR: 0.0000, Loss: 0.0075 Hint loss: 0.0000
Step 232600 (epoch 297.44), Elapsed: 2342.6 ms, LR: 0.0000, Loss: 0.0078 Hint loss: 0.0000
Step 232700 (epoch 297.57), Elapsed: 2361.8 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 232800 (epoch 297.70), Elapsed: 2383.1 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 232900 (epoch 297.83), Elapsed: 2355.7 ms, LR: 0.0000, Loss: 0.0075 Hint loss: 0.0000
Step 233000 (epoch 297.95), Elapsed: 2358.5 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000

[Validation] Epoch: 298.00, Elapsed: 720.4 ms, Error: 2.46

Step 233100 (epoch 298.08), Elapsed: 2359.4 ms, LR: 0.0000, Loss: 0.0081 Hint loss: 0.0000
Step 233200 (epoch 298.21), Elapsed: 2358.7 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 233300 (epoch 298.34), Elapsed: 2361.6 ms, LR: 0.0000, Loss: 0.0088 Hint loss: 0.0000
Step 233400 (epoch 298.47), Elapsed: 2349.3 ms, LR: 0.0000, Loss: 0.0084 Hint loss: 0.0000
Step 233500 (epoch 298.59), Elapsed: 2351.1 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 233600 (epoch 298.72), Elapsed: 2348.3 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 233700 (epoch 298.85), Elapsed: 2336.4 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 233800 (epoch 298.98), Elapsed: 2333.8 ms, LR: 0.0000, Loss: 0.0075 Hint loss: 0.0000

[Validation] Epoch: 299.00, Elapsed: 695.9 ms, Error: 2.51

Step 233900 (epoch 299.10), Elapsed: 2347.8 ms, LR: 0.0000, Loss: 0.0076 Hint loss: 0.0000
Step 234000 (epoch 299.23), Elapsed: 2341.9 ms, LR: 0.0000, Loss: 0.0079 Hint loss: 0.0000
Step 234100 (epoch 299.36), Elapsed: 2370.6 ms, LR: 0.0000, Loss: 0.0082 Hint loss: 0.0000
Step 234200 (epoch 299.49), Elapsed: 2388.1 ms, LR: 0.0000, Loss: 0.0075 Hint loss: 0.0000
Step 234300 (epoch 299.62), Elapsed: 2421.0 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000
Step 234400 (epoch 299.74), Elapsed: 2344.7 ms, LR: 0.0000, Loss: 0.0074 Hint loss: 0.0000
Step 234500 (epoch 299.87), Elapsed: 2358.0 ms, LR: 0.0000, Loss: 0.0077 Hint loss: 0.0000

[Validation] Epoch: 300.00, Elapsed: 705.3 ms, Error: 2.50
```

* Fine Model
```
Step 226800 (epoch 290.03), Elapsed: 9231.3 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 226900 (epoch 290.15), Elapsed: 9246.0 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 227000 (epoch 290.28), Elapsed: 9243.2 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 227100 (epoch 290.41), Elapsed: 9244.7 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 227200 (epoch 290.54), Elapsed: 9242.7 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 227300 (epoch 290.66), Elapsed: 9249.4 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 227400 (epoch 290.79), Elapsed: 9236.7 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 227500 (epoch 290.92), Elapsed: 9244.7 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000

[Validation] Epoch: 291.00, Elapsed: 4519.1 ms, Error: 1.05

Step 227600 (epoch 291.05), Elapsed: 9245.0 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 227700 (epoch 291.18), Elapsed: 9262.2 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 227800 (epoch 291.30), Elapsed: 9226.3 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 227900 (epoch 291.43), Elapsed: 9240.6 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 228000 (epoch 291.56), Elapsed: 9252.9 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 228100 (epoch 291.69), Elapsed: 9237.1 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 228200 (epoch 291.82), Elapsed: 9258.1 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 228300 (epoch 291.94), Elapsed: 9220.3 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000

[Validation] Epoch: 292.00, Elapsed: 5068.6 ms, Error: 1.10

Step 228400 (epoch 292.07), Elapsed: 9243.4 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 228500 (epoch 292.20), Elapsed: 9264.5 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 228600 (epoch 292.33), Elapsed: 9246.8 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 228700 (epoch 292.46), Elapsed: 9236.6 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 228800 (epoch 292.58), Elapsed: 9254.9 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 228900 (epoch 292.71), Elapsed: 9217.9 ms, LR: 0.0000, Loss: 0.0054 Hint loss: 0.0000
Step 229000 (epoch 292.84), Elapsed: 9242.8 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 229100 (epoch 292.97), Elapsed: 9263.0 ms, LR: 0.0000, Loss: 0.0043 Hint loss: 0.0000

[Validation] Epoch: 293.00, Elapsed: 5087.3 ms, Error: 1.10

Step 229200 (epoch 293.09), Elapsed: 9226.1 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 229300 (epoch 293.22), Elapsed: 9243.0 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 229400 (epoch 293.35), Elapsed: 9259.9 ms, LR: 0.0000, Loss: 0.0043 Hint loss: 0.0000
Step 229500 (epoch 293.48), Elapsed: 9234.5 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 229600 (epoch 293.61), Elapsed: 9268.0 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 229700 (epoch 293.73), Elapsed: 9247.7 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 229800 (epoch 293.86), Elapsed: 9233.1 ms, LR: 0.0000, Loss: 0.0044 Hint loss: 0.0000
Step 229900 (epoch 293.99), Elapsed: 9255.3 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000

[Validation] Epoch: 294.00, Elapsed: 4956.2 ms, Error: 1.05

Step 230000 (epoch 294.12), Elapsed: 9241.4 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 230100 (epoch 294.25), Elapsed: 9258.5 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 230200 (epoch 294.37), Elapsed: 9261.9 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 230300 (epoch 294.50), Elapsed: 9248.8 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 230400 (epoch 294.63), Elapsed: 9265.1 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 230500 (epoch 294.76), Elapsed: 9233.3 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 230600 (epoch 294.88), Elapsed: 9228.1 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000

[Validation] Epoch: 295.00, Elapsed: 4717.9 ms, Error: 1.09

Step 230700 (epoch 295.01), Elapsed: 9234.4 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 230800 (epoch 295.14), Elapsed: 9233.0 ms, LR: 0.0000, Loss: 0.0089 Hint loss: 0.0000
Step 230900 (epoch 295.27), Elapsed: 9231.4 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 231000 (epoch 295.40), Elapsed: 9242.6 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 231100 (epoch 295.52), Elapsed: 9236.5 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 231200 (epoch 295.65), Elapsed: 9225.5 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 231300 (epoch 295.78), Elapsed: 9256.8 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 231400 (epoch 295.91), Elapsed: 9251.2 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000

[Validation] Epoch: 296.00, Elapsed: 5093.4 ms, Error: 1.07

Step 231500 (epoch 296.04), Elapsed: 9252.2 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 231600 (epoch 296.16), Elapsed: 9241.5 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 231700 (epoch 296.29), Elapsed: 9277.3 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 231800 (epoch 296.42), Elapsed: 9243.7 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 231900 (epoch 296.55), Elapsed: 9261.0 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 232000 (epoch 296.68), Elapsed: 9276.0 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 232100 (epoch 296.80), Elapsed: 9261.5 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 232200 (epoch 296.93), Elapsed: 9264.5 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000

[Validation] Epoch: 297.00, Elapsed: 4994.9 ms, Error: 1.08

Step 232300 (epoch 297.06), Elapsed: 9242.4 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 232400 (epoch 297.19), Elapsed: 9254.1 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 232500 (epoch 297.31), Elapsed: 9223.2 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 232600 (epoch 297.44), Elapsed: 9240.1 ms, LR: 0.0000, Loss: 0.0043 Hint loss: 0.0000
Step 232700 (epoch 297.57), Elapsed: 9243.8 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 232800 (epoch 297.70), Elapsed: 9254.9 ms, LR: 0.0000, Loss: 0.0044 Hint loss: 0.0000
Step 232900 (epoch 297.83), Elapsed: 9240.3 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 233000 (epoch 297.95), Elapsed: 9262.9 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000

[Validation] Epoch: 298.00, Elapsed: 4263.9 ms, Error: 1.08

Step 233100 (epoch 298.08), Elapsed: 9243.2 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 233200 (epoch 298.21), Elapsed: 9225.3 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 233300 (epoch 298.34), Elapsed: 9251.7 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 233400 (epoch 298.47), Elapsed: 9246.1 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 233500 (epoch 298.59), Elapsed: 9269.8 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 233600 (epoch 298.72), Elapsed: 9267.0 ms, LR: 0.0000, Loss: 0.0043 Hint loss: 0.0000
Step 233700 (epoch 298.85), Elapsed: 9255.2 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 233800 (epoch 298.98), Elapsed: 9242.8 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000

[Validation] Epoch: 299.00, Elapsed: 4493.4 ms, Error: 1.06

Step 233900 (epoch 299.10), Elapsed: 9251.4 ms, LR: 0.0000, Loss: 0.0042 Hint loss: 0.0000
Step 234000 (epoch 299.23), Elapsed: 9240.5 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 234100 (epoch 299.36), Elapsed: 9243.3 ms, LR: 0.0000, Loss: 0.0039 Hint loss: 0.0000
Step 234200 (epoch 299.49), Elapsed: 9251.3 ms, LR: 0.0000, Loss: 0.0039 Hint loss: 0.0000
Step 234300 (epoch 299.62), Elapsed: 9228.4 ms, LR: 0.0000, Loss: 0.0041 Hint loss: 0.0000
Step 234400 (epoch 299.74), Elapsed: 9246.9 ms, LR: 0.0000, Loss: 0.0040 Hint loss: 0.0000
Step 234500 (epoch 299.87), Elapsed: 9248.9 ms, LR: 0.0000, Loss: 0.0039 Hint loss: 0.0000

[Validation] Epoch: 300.00, Elapsed: 4839.8 ms, Error: 1.02
```

* DCN without hint objective (8 patches)
```
Step 226800 (epoch 290.03), Elapsed: 8785.4 ms, LR: 0.0000, Loss: 0.0151 Hint loss: 140.5566
Step 226900 (epoch 290.15), Elapsed: 8795.0 ms, LR: 0.0000, Loss: 0.0159 Hint loss: 145.1720
Step 227000 (epoch 290.28), Elapsed: 8929.4 ms, LR: 0.0000, Loss: 0.0145 Hint loss: 152.0311
Step 227100 (epoch 290.41), Elapsed: 8850.0 ms, LR: 0.0000, Loss: 0.0169 Hint loss: 142.2562
Step 227200 (epoch 290.54), Elapsed: 8924.5 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 148.6252
Step 227300 (epoch 290.66), Elapsed: 8877.4 ms, LR: 0.0000, Loss: 0.0143 Hint loss: 142.6112
Step 227400 (epoch 290.79), Elapsed: 8599.6 ms, LR: 0.0000, Loss: 0.0224 Hint loss: 144.5831
Step 227500 (epoch 290.92), Elapsed: 8886.5 ms, LR: 0.0000, Loss: 0.0147 Hint loss: 142.7310

[Validation] Epoch: 291.00, Elapsed: 10625.1 ms, Error: 1.38

Step 227600 (epoch 291.05), Elapsed: 8960.8 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 153.9186
Step 227700 (epoch 291.18), Elapsed: 8787.3 ms, LR: 0.0000, Loss: 0.0141 Hint loss: 153.5106
Step 227800 (epoch 291.30), Elapsed: 8964.0 ms, LR: 0.0000, Loss: 0.0144 Hint loss: 146.8593
Step 227900 (epoch 291.43), Elapsed: 8749.9 ms, LR: 0.0000, Loss: 0.0165 Hint loss: 139.4108
Step 228000 (epoch 291.56), Elapsed: 8837.5 ms, LR: 0.0000, Loss: 0.0141 Hint loss: 140.7742
Step 228100 (epoch 291.69), Elapsed: 8907.0 ms, LR: 0.0000, Loss: 0.0153 Hint loss: 139.0107
Step 228200 (epoch 291.82), Elapsed: 8989.3 ms, LR: 0.0000, Loss: 0.0173 Hint loss: 139.3120
Step 228300 (epoch 291.94), Elapsed: 8778.1 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 152.0138

[Validation] Epoch: 292.00, Elapsed: 10245.9 ms, Error: 1.44

Step 228400 (epoch 292.07), Elapsed: 8994.7 ms, LR: 0.0000, Loss: 0.0143 Hint loss: 144.2561
Step 228500 (epoch 292.20), Elapsed: 8811.0 ms, LR: 0.0000, Loss: 0.0141 Hint loss: 136.9738
Step 228600 (epoch 292.33), Elapsed: 8843.2 ms, LR: 0.0000, Loss: 0.0149 Hint loss: 149.8417
Step 228700 (epoch 292.46), Elapsed: 8781.2 ms, LR: 0.0000, Loss: 0.0155 Hint loss: 153.1330
Step 228800 (epoch 292.58), Elapsed: 8839.1 ms, LR: 0.0000, Loss: 0.0164 Hint loss: 151.2229
Step 228900 (epoch 292.71), Elapsed: 8861.1 ms, LR: 0.0000, Loss: 0.0175 Hint loss: 142.4727
Step 229000 (epoch 292.84), Elapsed: 8883.7 ms, LR: 0.0000, Loss: 0.0148 Hint loss: 141.8695
Step 229100 (epoch 292.97), Elapsed: 8808.3 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 144.9092

[Validation] Epoch: 293.00, Elapsed: 10523.7 ms, Error: 1.42

Step 229200 (epoch 293.09), Elapsed: 8858.2 ms, LR: 0.0000, Loss: 0.0149 Hint loss: 149.5001
Step 229300 (epoch 293.22), Elapsed: 8811.4 ms, LR: 0.0000, Loss: 0.0176 Hint loss: 143.4723
Step 229400 (epoch 293.35), Elapsed: 9023.6 ms, LR: 0.0000, Loss: 0.0162 Hint loss: 144.8050
Step 229500 (epoch 293.48), Elapsed: 8871.7 ms, LR: 0.0000, Loss: 0.0147 Hint loss: 145.9355
Step 229600 (epoch 293.61), Elapsed: 8747.7 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 143.7787
Step 229700 (epoch 293.73), Elapsed: 8809.1 ms, LR: 0.0000, Loss: 0.0169 Hint loss: 140.3045
Step 229800 (epoch 293.86), Elapsed: 8920.7 ms, LR: 0.0000, Loss: 0.0146 Hint loss: 146.5044
Step 229900 (epoch 293.99), Elapsed: 8758.8 ms, LR: 0.0000, Loss: 0.0161 Hint loss: 143.6587

[Validation] Epoch: 294.00, Elapsed: 10459.7 ms, Error: 1.38

Step 230000 (epoch 294.12), Elapsed: 8855.6 ms, LR: 0.0000, Loss: 0.0143 Hint loss: 148.2706
Step 230100 (epoch 294.25), Elapsed: 8912.5 ms, LR: 0.0000, Loss: 0.0168 Hint loss: 142.8831
Step 230200 (epoch 294.37), Elapsed: 8878.7 ms, LR: 0.0000, Loss: 0.0141 Hint loss: 151.0914
Step 230300 (epoch 294.50), Elapsed: 8739.2 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 146.8802
Step 230400 (epoch 294.63), Elapsed: 8797.3 ms, LR: 0.0000, Loss: 0.0145 Hint loss: 144.5435
Step 230500 (epoch 294.76), Elapsed: 8922.8 ms, LR: 0.0000, Loss: 0.0151 Hint loss: 138.8512
Step 230600 (epoch 294.88), Elapsed: 8798.5 ms, LR: 0.0000, Loss: 0.0144 Hint loss: 147.0982

[Validation] Epoch: 295.00, Elapsed: 10150.6 ms, Error: 1.50

Step 230700 (epoch 295.01), Elapsed: 8841.0 ms, LR: 0.0000, Loss: 0.0144 Hint loss: 150.0737
Step 230800 (epoch 295.14), Elapsed: 8830.3 ms, LR: 0.0000, Loss: 0.0145 Hint loss: 148.4242
Step 230900 (epoch 295.27), Elapsed: 8895.8 ms, LR: 0.0000, Loss: 0.0139 Hint loss: 142.8199
Step 231000 (epoch 295.40), Elapsed: 8825.8 ms, LR: 0.0000, Loss: 0.0141 Hint loss: 147.1053
Step 231100 (epoch 295.52), Elapsed: 8746.1 ms, LR: 0.0000, Loss: 0.0154 Hint loss: 145.1556
Step 231200 (epoch 295.65), Elapsed: 8861.3 ms, LR: 0.0000, Loss: 0.0146 Hint loss: 150.5138
Step 231300 (epoch 295.78), Elapsed: 8949.1 ms, LR: 0.0000, Loss: 0.0141 Hint loss: 152.0608
Step 231400 (epoch 295.91), Elapsed: 8834.2 ms, LR: 0.0000, Loss: 0.0145 Hint loss: 142.8384

[Validation] Epoch: 296.00, Elapsed: 10195.9 ms, Error: 1.41

Step 231500 (epoch 296.04), Elapsed: 8922.9 ms, LR: 0.0000, Loss: 0.0156 Hint loss: 151.1056
Step 231600 (epoch 296.16), Elapsed: 8919.9 ms, LR: 0.0000, Loss: 0.0148 Hint loss: 156.8548
Step 231700 (epoch 296.29), Elapsed: 8975.1 ms, LR: 0.0000, Loss: 0.0143 Hint loss: 150.7839
Step 231800 (epoch 296.42), Elapsed: 8864.1 ms, LR: 0.0000, Loss: 0.0153 Hint loss: 141.7013
Step 231900 (epoch 296.55), Elapsed: 8857.3 ms, LR: 0.0000, Loss: 0.0141 Hint loss: 142.0977
Step 232000 (epoch 296.68), Elapsed: 8978.7 ms, LR: 0.0000, Loss: 0.0148 Hint loss: 152.2045
Step 232100 (epoch 296.80), Elapsed: 8791.5 ms, LR: 0.0000, Loss: 0.0169 Hint loss: 145.3065
Step 232200 (epoch 296.93), Elapsed: 8822.5 ms, LR: 0.0000, Loss: 0.0144 Hint loss: 157.7260

[Validation] Epoch: 297.00, Elapsed: 9870.3 ms, Error: 1.51

Step 232300 (epoch 297.06), Elapsed: 8903.2 ms, LR: 0.0000, Loss: 0.0149 Hint loss: 145.8231
Step 232400 (epoch 297.19), Elapsed: 8807.7 ms, LR: 0.0000, Loss: 0.0139 Hint loss: 145.0216
Step 232500 (epoch 297.31), Elapsed: 9031.0 ms, LR: 0.0000, Loss: 0.0143 Hint loss: 142.4312
Step 232600 (epoch 297.44), Elapsed: 8765.8 ms, LR: 0.0000, Loss: 0.0139 Hint loss: 153.0089
Step 232700 (epoch 297.57), Elapsed: 8888.1 ms, LR: 0.0000, Loss: 0.0148 Hint loss: 143.9904
Step 232800 (epoch 297.70), Elapsed: 8753.2 ms, LR: 0.0000, Loss: 0.0144 Hint loss: 149.5293
Step 232900 (epoch 297.83), Elapsed: 8880.4 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 152.8657
Step 233000 (epoch 297.95), Elapsed: 8774.6 ms, LR: 0.0000, Loss: 0.0152 Hint loss: 145.9120

[Validation] Epoch: 298.00, Elapsed: 10043.2 ms, Error: 1.39

Step 233100 (epoch 298.08), Elapsed: 8851.3 ms, LR: 0.0000, Loss: 0.0140 Hint loss: 146.8957
Step 233200 (epoch 298.21), Elapsed: 8814.8 ms, LR: 0.0000, Loss: 0.0139 Hint loss: 146.9237
Step 233300 (epoch 298.34), Elapsed: 8897.6 ms, LR: 0.0000, Loss: 0.0141 Hint loss: 151.6163
Step 233400 (epoch 298.47), Elapsed: 8875.2 ms, LR: 0.0000, Loss: 0.0143 Hint loss: 149.1765
Step 233500 (epoch 298.59), Elapsed: 8778.8 ms, LR: 0.0000, Loss: 0.0145 Hint loss: 139.9828
Step 233600 (epoch 298.72), Elapsed: 8786.9 ms, LR: 0.0000, Loss: 0.0146 Hint loss: 140.4219
Step 233700 (epoch 298.85), Elapsed: 8716.7 ms, LR: 0.0000, Loss: 0.0144 Hint loss: 146.3438
Step 233800 (epoch 298.98), Elapsed: 8868.1 ms, LR: 0.0000, Loss: 0.0147 Hint loss: 143.4257

[Validation] Epoch: 299.00, Elapsed: 10164.8 ms, Error: 1.46

Step 233900 (epoch 299.10), Elapsed: 8752.9 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 150.7824
Step 234000 (epoch 299.23), Elapsed: 8731.2 ms, LR: 0.0000, Loss: 0.0139 Hint loss: 149.9018
Step 234100 (epoch 299.36), Elapsed: 8959.3 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 143.6503
Step 234200 (epoch 299.49), Elapsed: 8808.8 ms, LR: 0.0000, Loss: 0.0140 Hint loss: 148.8103
Step 234300 (epoch 299.62), Elapsed: 8818.4 ms, LR: 0.0000, Loss: 0.0140 Hint loss: 154.3216
Step 234400 (epoch 299.74), Elapsed: 8995.8 ms, LR: 0.0000, Loss: 0.0140 Hint loss: 143.6152
Step 234500 (epoch 299.87), Elapsed: 8852.7 ms, LR: 0.0000, Loss: 0.0148 Hint loss: 156.2379

[Validation] Epoch: 300.00, Elapsed: 9840.2 ms, Error: 1.49
```

* DCN with hint objective (8 patches)
```
Step 226800 (epoch 290.03), Elapsed: 8942.4 ms, LR: 0.0000, Loss: 0.0116 Hint loss: 4.8372
Step 226900 (epoch 290.15), Elapsed: 8846.7 ms, LR: 0.0000, Loss: 0.0116 Hint loss: 4.8114
Step 227000 (epoch 290.28), Elapsed: 8904.0 ms, LR: 0.0000, Loss: 0.0123 Hint loss: 5.1334
Step 227100 (epoch 290.41), Elapsed: 8825.9 ms, LR: 0.0000, Loss: 0.0127 Hint loss: 4.9805
Step 227200 (epoch 290.54), Elapsed: 8984.5 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 4.9500
Step 227300 (epoch 290.66), Elapsed: 8848.2 ms, LR: 0.0000, Loss: 0.0122 Hint loss: 4.9223
Step 227400 (epoch 290.79), Elapsed: 8759.7 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 5.2011
Step 227500 (epoch 290.92), Elapsed: 9105.3 ms, LR: 0.0000, Loss: 0.0121 Hint loss: 4.9261

[Validation] Epoch: 291.00, Elapsed: 10333.4 ms, Error: 1.19

Step 227600 (epoch 291.05), Elapsed: 8958.3 ms, LR: 0.0000, Loss: 0.0139 Hint loss: 4.9091
Step 227700 (epoch 291.18), Elapsed: 8987.6 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 4.7313
Step 227800 (epoch 291.30), Elapsed: 8957.2 ms, LR: 0.0000, Loss: 0.0119 Hint loss: 4.7831
Step 227900 (epoch 291.43), Elapsed: 8949.5 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 4.8308
Step 228000 (epoch 291.56), Elapsed: 8950.6 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 4.7272
Step 228100 (epoch 291.69), Elapsed: 8726.6 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 4.9173
Step 228200 (epoch 291.82), Elapsed: 8832.2 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 5.1625
Step 228300 (epoch 291.94), Elapsed: 8973.3 ms, LR: 0.0000, Loss: 0.0131 Hint loss: 5.1492

[Validation] Epoch: 292.00, Elapsed: 10217.6 ms, Error: 1.17

Step 228400 (epoch 292.07), Elapsed: 8842.0 ms, LR: 0.0000, Loss: 0.0117 Hint loss: 5.1325
Step 228500 (epoch 292.20), Elapsed: 8917.8 ms, LR: 0.0000, Loss: 0.0122 Hint loss: 4.9998
Step 228600 (epoch 292.33), Elapsed: 8969.1 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 5.0996
Step 228700 (epoch 292.46), Elapsed: 8847.0 ms, LR: 0.0000, Loss: 0.0117 Hint loss: 5.1582
Step 228800 (epoch 292.58), Elapsed: 8815.3 ms, LR: 0.0000, Loss: 0.0146 Hint loss: 4.7797
Step 228900 (epoch 292.71), Elapsed: 8812.4 ms, LR: 0.0000, Loss: 0.0119 Hint loss: 4.9511
Step 229000 (epoch 292.84), Elapsed: 8860.3 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 4.9879
Step 229100 (epoch 292.97), Elapsed: 8911.2 ms, LR: 0.0000, Loss: 0.0156 Hint loss: 4.9679

[Validation] Epoch: 293.00, Elapsed: 10359.0 ms, Error: 1.20

Step 229200 (epoch 293.09), Elapsed: 8857.3 ms, LR: 0.0000, Loss: 0.0122 Hint loss: 5.1489
Step 229300 (epoch 293.22), Elapsed: 8794.5 ms, LR: 0.0000, Loss: 0.0120 Hint loss: 4.7278
Step 229400 (epoch 293.35), Elapsed: 8760.3 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 5.2000
Step 229500 (epoch 293.48), Elapsed: 8779.9 ms, LR: 0.0000, Loss: 0.0151 Hint loss: 4.8955
Step 229600 (epoch 293.61), Elapsed: 8799.6 ms, LR: 0.0000, Loss: 0.0142 Hint loss: 5.3606
Step 229700 (epoch 293.73), Elapsed: 8842.7 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 4.7102
Step 229800 (epoch 293.86), Elapsed: 9056.3 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 4.7298
Step 229900 (epoch 293.99), Elapsed: 8871.5 ms, LR: 0.0000, Loss: 0.0116 Hint loss: 4.9709

[Validation] Epoch: 294.00, Elapsed: 10239.3 ms, Error: 1.21

Step 230000 (epoch 294.12), Elapsed: 8889.2 ms, LR: 0.0000, Loss: 0.0116 Hint loss: 4.9472
Step 230100 (epoch 294.25), Elapsed: 8955.4 ms, LR: 0.0000, Loss: 0.0123 Hint loss: 4.9136
Step 230200 (epoch 294.37), Elapsed: 8938.4 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 5.1700
Step 230300 (epoch 294.50), Elapsed: 8775.8 ms, LR: 0.0000, Loss: 0.0133 Hint loss: 5.0109
Step 230400 (epoch 294.63), Elapsed: 8863.7 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 4.9127
Step 230500 (epoch 294.76), Elapsed: 8881.8 ms, LR: 0.0000, Loss: 0.0122 Hint loss: 4.9475
Step 230600 (epoch 294.88), Elapsed: 8937.7 ms, LR: 0.0000, Loss: 0.0126 Hint loss: 4.6613

[Validation] Epoch: 295.00, Elapsed: 10070.9 ms, Error: 1.24

Step 230700 (epoch 295.01), Elapsed: 8838.6 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 5.2808
Step 230800 (epoch 295.14), Elapsed: 8875.8 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 5.1866
Step 230900 (epoch 295.27), Elapsed: 8897.8 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 4.8991
Step 231000 (epoch 295.40), Elapsed: 8976.7 ms, LR: 0.0000, Loss: 0.0116 Hint loss: 4.8758
Step 231100 (epoch 295.52), Elapsed: 8984.5 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 5.0400
Step 231200 (epoch 295.65), Elapsed: 8862.5 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 5.5212
Step 231300 (epoch 295.78), Elapsed: 8803.1 ms, LR: 0.0000, Loss: 0.0120 Hint loss: 5.0450
Step 231400 (epoch 295.91), Elapsed: 8881.1 ms, LR: 0.0000, Loss: 0.0117 Hint loss: 5.0264

[Validation] Epoch: 296.00, Elapsed: 10189.6 ms, Error: 1.16

Step 231500 (epoch 296.04), Elapsed: 8924.5 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 4.7068
Step 231600 (epoch 296.16), Elapsed: 8846.3 ms, LR: 0.0000, Loss: 0.0120 Hint loss: 4.7155
Step 231700 (epoch 296.29), Elapsed: 8882.1 ms, LR: 0.0000, Loss: 0.0119 Hint loss: 5.1504
Step 231800 (epoch 296.42), Elapsed: 8829.3 ms, LR: 0.0000, Loss: 0.0119 Hint loss: 4.7586
Step 231900 (epoch 296.55), Elapsed: 8883.2 ms, LR: 0.0000, Loss: 0.0116 Hint loss: 4.7570
Step 232000 (epoch 296.68), Elapsed: 8799.7 ms, LR: 0.0000, Loss: 0.0177 Hint loss: 5.1904
Step 232100 (epoch 296.80), Elapsed: 9038.5 ms, LR: 0.0000, Loss: 0.0135 Hint loss: 4.6850
Step 232200 (epoch 296.93), Elapsed: 8823.2 ms, LR: 0.0000, Loss: 0.0113 Hint loss: 4.5412

[Validation] Epoch: 297.00, Elapsed: 10070.1 ms, Error: 1.14

Step 232300 (epoch 297.06), Elapsed: 9001.3 ms, LR: 0.0000, Loss: 0.0113 Hint loss: 4.7578
Step 232400 (epoch 297.19), Elapsed: 8940.8 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 4.8155
Step 232500 (epoch 297.31), Elapsed: 8797.1 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 4.9381
Step 232600 (epoch 297.44), Elapsed: 8870.4 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 4.9699
Step 232700 (epoch 297.57), Elapsed: 8864.3 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 5.2074
Step 232800 (epoch 297.70), Elapsed: 8857.6 ms, LR: 0.0000, Loss: 0.0113 Hint loss: 4.8059
Step 232900 (epoch 297.83), Elapsed: 9129.8 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 4.9088
Step 233000 (epoch 297.95), Elapsed: 8834.6 ms, LR: 0.0000, Loss: 0.0116 Hint loss: 4.9354

[Validation] Epoch: 298.00, Elapsed: 10232.5 ms, Error: 1.20

Step 233100 (epoch 298.08), Elapsed: 8874.6 ms, LR: 0.0000, Loss: 0.0124 Hint loss: 4.7468
Step 233200 (epoch 298.21), Elapsed: 8927.0 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 4.8656
Step 233300 (epoch 298.34), Elapsed: 8934.8 ms, LR: 0.0000, Loss: 0.0113 Hint loss: 4.9476
Step 233400 (epoch 298.47), Elapsed: 8970.3 ms, LR: 0.0000, Loss: 0.0120 Hint loss: 4.6907
Step 233500 (epoch 298.59), Elapsed: 8850.5 ms, LR: 0.0000, Loss: 0.0115 Hint loss: 5.1286
Step 233600 (epoch 298.72), Elapsed: 8806.0 ms, LR: 0.0000, Loss: 0.0117 Hint loss: 5.3058
Step 233700 (epoch 298.85), Elapsed: 8785.5 ms, LR: 0.0000, Loss: 0.0117 Hint loss: 4.9079
Step 233800 (epoch 298.98), Elapsed: 8005.3 ms, LR: 0.0000, Loss: 0.0143 Hint loss: 4.9513

[Validation] Epoch: 299.00, Elapsed: 8645.1 ms, Error: 1.26

Step 233900 (epoch 299.10), Elapsed: 7949.9 ms, LR: 0.0000, Loss: 0.0117 Hint loss: 4.9189
Step 234000 (epoch 299.23), Elapsed: 7985.1 ms, LR: 0.0000, Loss: 0.0117 Hint loss: 5.0969
Step 234100 (epoch 299.36), Elapsed: 8105.3 ms, LR: 0.0000, Loss: 0.0118 Hint loss: 5.0055
Step 234200 (epoch 299.49), Elapsed: 8025.6 ms, LR: 0.0000, Loss: 0.0113 Hint loss: 4.7879
Step 234300 (epoch 299.62), Elapsed: 8173.1 ms, LR: 0.0000, Loss: 0.0114 Hint loss: 4.6951
Step 234400 (epoch 299.74), Elapsed: 8081.3 ms, LR: 0.0000, Loss: 0.0124 Hint loss: 4.9987
Step 234500 (epoch 299.87), Elapsed: 8046.9 ms, LR: 0.0000, Loss: 0.0113 Hint loss: 5.0564

[Validation] Epoch: 300.00, Elapsed: 8688.6 ms, Error: 1.20
```
