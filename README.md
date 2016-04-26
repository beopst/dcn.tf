# Dynamic Capacity Networks using Tensorflow

Dynamic Capacity Networks (DCN; http://arxiv.org/abs/1511.07838) implementation using Tensorflow. DCN reduces the number of computations by applying high-capacity networks to some selected input patches. 

## Dataset

* [Cluttered MNIST](https://github.com/deepmind/mnist-cluttered)

## NOTES

* It is observed that DCN outperforms Coarse Model. However, it is still worse than Fine Model regardless of using hint objective. Hint objective gives us slightly (or insignificant) better generalization.

* In this benchmark, batch normalization was applied to only convolution layers. Weight decay parameter was set to 0.001, and additional weight parameter for hint objective was set to 0.01.

* Adam optimizer was used. Initial learning rate was 0.001, and it was decreased a factor of 10 for every 20 epochs. Total number of epochs was 50. 

* Computation time issue
>Although DCN can greatly reduce the number of computations, it may consume more computation time since it requires sequantial processes (extract input patches of interest, and then feed them into high-capacity networks). Computation time can be reduced if the input patches are concatenated for convolution.

## Logs
* Coarse Model
```
[Validation] Epoch: 40.00, Elapsed: 695.5 ms, Error: 2.45

Step 31300 (epoch 40.03), Elapsed: 2390.5 ms, LR: 0.0000, Loss: 0.0208 Hint loss: 0.0000
Step 31400 (epoch 40.15), Elapsed: 2370.4 ms, LR: 0.0000, Loss: 0.0214 Hint loss: 0.0000
Step 31500 (epoch 40.28), Elapsed: 2374.6 ms, LR: 0.0000, Loss: 0.0225 Hint loss: 0.0000
Step 31600 (epoch 40.41), Elapsed: 2373.3 ms, LR: 0.0000, Loss: 0.0207 Hint loss: 0.0000
Step 31700 (epoch 40.54), Elapsed: 2380.1 ms, LR: 0.0000, Loss: 0.0202 Hint loss: 0.0000
Step 31800 (epoch 40.66), Elapsed: 2376.3 ms, LR: 0.0000, Loss: 0.0229 Hint loss: 0.0000
Step 31900 (epoch 40.79), Elapsed: 2379.9 ms, LR: 0.0000, Loss: 0.0200 Hint loss: 0.0000
Step 32000 (epoch 40.92), Elapsed: 2376.8 ms, LR: 0.0000, Loss: 0.0232 Hint loss: 0.0000

[Validation] Epoch: 41.00, Elapsed: 685.5 ms, Error: 2.40

Step 32100 (epoch 41.05), Elapsed: 2387.3 ms, LR: 0.0000, Loss: 0.0209 Hint loss: 0.0000
Step 32200 (epoch 41.18), Elapsed: 2373.0 ms, LR: 0.0000, Loss: 0.0205 Hint loss: 0.0000
Step 32300 (epoch 41.30), Elapsed: 2378.5 ms, LR: 0.0000, Loss: 0.0201 Hint loss: 0.0000
Step 32400 (epoch 41.43), Elapsed: 2383.1 ms, LR: 0.0000, Loss: 0.0206 Hint loss: 0.0000
Step 32500 (epoch 41.56), Elapsed: 2381.1 ms, LR: 0.0000, Loss: 0.0219 Hint loss: 0.0000
Step 32600 (epoch 41.69), Elapsed: 2383.1 ms, LR: 0.0000, Loss: 0.0211 Hint loss: 0.0000
Step 32700 (epoch 41.82), Elapsed: 2404.9 ms, LR: 0.0000, Loss: 0.0210 Hint loss: 0.0000
Step 32800 (epoch 41.94), Elapsed: 2400.4 ms, LR: 0.0000, Loss: 0.0216 Hint loss: 0.0000

[Validation] Epoch: 42.00, Elapsed: 697.1 ms, Error: 2.39

Step 32900 (epoch 42.07), Elapsed: 2382.6 ms, LR: 0.0000, Loss: 0.0209 Hint loss: 0.0000
Step 33000 (epoch 42.20), Elapsed: 2364.3 ms, LR: 0.0000, Loss: 0.0200 Hint loss: 0.0000
Step 33100 (epoch 42.33), Elapsed: 2373.7 ms, LR: 0.0000, Loss: 0.0202 Hint loss: 0.0000
Step 33200 (epoch 42.46), Elapsed: 2373.6 ms, LR: 0.0000, Loss: 0.0205 Hint loss: 0.0000
Step 33300 (epoch 42.58), Elapsed: 2381.7 ms, LR: 0.0000, Loss: 0.0203 Hint loss: 0.0000
Step 33400 (epoch 42.71), Elapsed: 2380.1 ms, LR: 0.0000, Loss: 0.0200 Hint loss: 0.0000
Step 33500 (epoch 42.84), Elapsed: 2371.7 ms, LR: 0.0000, Loss: 0.0200 Hint loss: 0.0000
Step 33600 (epoch 42.97), Elapsed: 2392.8 ms, LR: 0.0000, Loss: 0.0224 Hint loss: 0.0000

[Validation] Epoch: 43.00, Elapsed: 684.8 ms, Error: 2.31

Step 33700 (epoch 43.09), Elapsed: 2384.4 ms, LR: 0.0000, Loss: 0.0207 Hint loss: 0.0000
Step 33800 (epoch 43.22), Elapsed: 2371.1 ms, LR: 0.0000, Loss: 0.0207 Hint loss: 0.0000
Step 33900 (epoch 43.35), Elapsed: 2374.6 ms, LR: 0.0000, Loss: 0.0201 Hint loss: 0.0000
Step 34000 (epoch 43.48), Elapsed: 2377.6 ms, LR: 0.0000, Loss: 0.0198 Hint loss: 0.0000
Step 34100 (epoch 43.61), Elapsed: 2377.1 ms, LR: 0.0000, Loss: 0.0220 Hint loss: 0.0000
Step 34200 (epoch 43.73), Elapsed: 2381.6 ms, LR: 0.0000, Loss: 0.0203 Hint loss: 0.0000
Step 34300 (epoch 43.86), Elapsed: 2390.3 ms, LR: 0.0000, Loss: 0.0209 Hint loss: 0.0000
Step 34400 (epoch 43.99), Elapsed: 2403.7 ms, LR: 0.0000, Loss: 0.0199 Hint loss: 0.0000

[Validation] Epoch: 44.00, Elapsed: 691.8 ms, Error: 2.30

Step 34500 (epoch 44.12), Elapsed: 2394.6 ms, LR: 0.0000, Loss: 0.0196 Hint loss: 0.0000
Step 34600 (epoch 44.25), Elapsed: 2379.5 ms, LR: 0.0000, Loss: 0.0204 Hint loss: 0.0000
Step 34700 (epoch 44.37), Elapsed: 2385.3 ms, LR: 0.0000, Loss: 0.0199 Hint loss: 0.0000
Step 34800 (epoch 44.50), Elapsed: 2375.8 ms, LR: 0.0000, Loss: 0.0197 Hint loss: 0.0000
Step 34900 (epoch 44.63), Elapsed: 2375.5 ms, LR: 0.0000, Loss: 0.0196 Hint loss: 0.0000
Step 35000 (epoch 44.76), Elapsed: 2375.6 ms, LR: 0.0000, Loss: 0.0201 Hint loss: 0.0000
Step 35100 (epoch 44.88), Elapsed: 2383.3 ms, LR: 0.0000, Loss: 0.0213 Hint loss: 0.0000

[Validation] Epoch: 45.00, Elapsed: 686.0 ms, Error: 2.36

Step 35200 (epoch 45.01), Elapsed: 2381.1 ms, LR: 0.0000, Loss: 0.0199 Hint loss: 0.0000
Step 35300 (epoch 45.14), Elapsed: 2367.0 ms, LR: 0.0000, Loss: 0.0219 Hint loss: 0.0000
Step 35400 (epoch 45.27), Elapsed: 2365.4 ms, LR: 0.0000, Loss: 0.0201 Hint loss: 0.0000
Step 35500 (epoch 45.40), Elapsed: 2373.8 ms, LR: 0.0000, Loss: 0.0196 Hint loss: 0.0000
Step 35600 (epoch 45.52), Elapsed: 2373.4 ms, LR: 0.0000, Loss: 0.0202 Hint loss: 0.0000
Step 35700 (epoch 45.65), Elapsed: 2404.5 ms, LR: 0.0000, Loss: 0.0214 Hint loss: 0.0000
Step 35800 (epoch 45.78), Elapsed: 2399.5 ms, LR: 0.0000, Loss: 0.0202 Hint loss: 0.0000
Step 35900 (epoch 45.91), Elapsed: 2402.0 ms, LR: 0.0000, Loss: 0.0194 Hint loss: 0.0000

[Validation] Epoch: 46.00, Elapsed: 709.7 ms, Error: 2.35

Step 36000 (epoch 46.04), Elapsed: 2390.5 ms, LR: 0.0000, Loss: 0.0196 Hint loss: 0.0000
Step 36100 (epoch 46.16), Elapsed: 2370.9 ms, LR: 0.0000, Loss: 0.0199 Hint loss: 0.0000
Step 36200 (epoch 46.29), Elapsed: 2375.7 ms, LR: 0.0000, Loss: 0.0200 Hint loss: 0.0000
Step 36300 (epoch 46.42), Elapsed: 2356.7 ms, LR: 0.0000, Loss: 0.0191 Hint loss: 0.0000
Step 36400 (epoch 46.55), Elapsed: 2361.4 ms, LR: 0.0000, Loss: 0.0196 Hint loss: 0.0000
Step 36500 (epoch 46.68), Elapsed: 2369.0 ms, LR: 0.0000, Loss: 0.0192 Hint loss: 0.0000
Step 36600 (epoch 46.80), Elapsed: 2368.6 ms, LR: 0.0000, Loss: 0.0199 Hint loss: 0.0000
Step 36700 (epoch 46.93), Elapsed: 2382.7 ms, LR: 0.0000, Loss: 0.0194 Hint loss: 0.0000

[Validation] Epoch: 47.00, Elapsed: 700.7 ms, Error: 2.34

Step 36800 (epoch 47.06), Elapsed: 2382.1 ms, LR: 0.0000, Loss: 0.0192 Hint loss: 0.0000
Step 36900 (epoch 47.19), Elapsed: 2377.8 ms, LR: 0.0000, Loss: 0.0196 Hint loss: 0.0000
Step 37000 (epoch 47.31), Elapsed: 2369.6 ms, LR: 0.0000, Loss: 0.0204 Hint loss: 0.0000
Step 37100 (epoch 47.44), Elapsed: 2381.0 ms, LR: 0.0000, Loss: 0.0189 Hint loss: 0.0000
Step 37200 (epoch 47.57), Elapsed: 2380.8 ms, LR: 0.0000, Loss: 0.0192 Hint loss: 0.0000
Step 37300 (epoch 47.70), Elapsed: 2375.5 ms, LR: 0.0000, Loss: 0.0201 Hint loss: 0.0000
Step 37400 (epoch 47.83), Elapsed: 2379.6 ms, LR: 0.0000, Loss: 0.0193 Hint loss: 0.0000
Step 37500 (epoch 47.95), Elapsed: 2376.7 ms, LR: 0.0000, Loss: 0.0188 Hint loss: 0.0000

[Validation] Epoch: 48.00, Elapsed: 682.2 ms, Error: 2.34

Step 37600 (epoch 48.08), Elapsed: 2379.7 ms, LR: 0.0000, Loss: 0.0197 Hint loss: 0.0000
Step 37700 (epoch 48.21), Elapsed: 2379.8 ms, LR: 0.0000, Loss: 0.0192 Hint loss: 0.0000
Step 37800 (epoch 48.34), Elapsed: 2377.5 ms, LR: 0.0000, Loss: 0.0192 Hint loss: 0.0000
Step 37900 (epoch 48.47), Elapsed: 2388.0 ms, LR: 0.0000, Loss: 0.0202 Hint loss: 0.0000
Step 38000 (epoch 48.59), Elapsed: 2385.4 ms, LR: 0.0000, Loss: 0.0194 Hint loss: 0.0000
Step 38100 (epoch 48.72), Elapsed: 2388.1 ms, LR: 0.0000, Loss: 0.0195 Hint loss: 0.0000
Step 38200 (epoch 48.85), Elapsed: 2373.1 ms, LR: 0.0000, Loss: 0.0189 Hint loss: 0.0000
Step 38300 (epoch 48.98), Elapsed: 2348.6 ms, LR: 0.0000, Loss: 0.0184 Hint loss: 0.0000

[Validation] Epoch: 49.00, Elapsed: 647.7 ms, Error: 2.39

Step 38400 (epoch 49.10), Elapsed: 2343.7 ms, LR: 0.0000, Loss: 0.0189 Hint loss: 0.0000
Step 38500 (epoch 49.23), Elapsed: 2343.3 ms, LR: 0.0000, Loss: 0.0200 Hint loss: 0.0000
Step 38600 (epoch 49.36), Elapsed: 2348.3 ms, LR: 0.0000, Loss: 0.0195 Hint loss: 0.0000
Step 38700 (epoch 49.49), Elapsed: 2367.9 ms, LR: 0.0000, Loss: 0.0194 Hint loss: 0.0000
Step 38800 (epoch 49.62), Elapsed: 2373.8 ms, LR: 0.0000, Loss: 0.0193 Hint loss: 0.0000
Step 38900 (epoch 49.74), Elapsed: 2372.7 ms, LR: 0.0000, Loss: 0.0199 Hint loss: 0.0000
Step 39000 (epoch 49.87), Elapsed: 2371.0 ms, LR: 0.0000, Loss: 0.0191 Hint loss: 0.0000

[Validation] Epoch: 50.00, Elapsed: 647.4 ms, Error: 2.32

Step 39100 (epoch 50.00), Elapsed: 2376.8 ms, LR: 0.0000, Loss: 0.0186 Hint loss: 0.0000
```

* Fine Model
```
[Validation] Epoch: 40.00, Elapsed: 2667.5 ms, Error: 1.24

Step 31300 (epoch 40.03), Elapsed: 8900.9 ms, LR: 0.0000, Loss: 0.0191 Hint loss: 0.0000
Step 31400 (epoch 40.15), Elapsed: 8892.3 ms, LR: 0.0000, Loss: 0.0191 Hint loss: 0.0000
Step 31500 (epoch 40.28), Elapsed: 8904.2 ms, LR: 0.0000, Loss: 0.0186 Hint loss: 0.0000
Step 31600 (epoch 40.41), Elapsed: 8893.0 ms, LR: 0.0000, Loss: 0.0191 Hint loss: 0.0000
Step 31700 (epoch 40.54), Elapsed: 8892.1 ms, LR: 0.0000, Loss: 0.0186 Hint loss: 0.0000
Step 31800 (epoch 40.66), Elapsed: 8899.7 ms, LR: 0.0000, Loss: 0.0189 Hint loss: 0.0000
Step 31900 (epoch 40.79), Elapsed: 8895.4 ms, LR: 0.0000, Loss: 0.0188 Hint loss: 0.0000
Step 32000 (epoch 40.92), Elapsed: 8898.6 ms, LR: 0.0000, Loss: 0.0195 Hint loss: 0.0000

[Validation] Epoch: 41.00, Elapsed: 2675.6 ms, Error: 1.17

Step 32100 (epoch 41.05), Elapsed: 8900.6 ms, LR: 0.0000, Loss: 0.0188 Hint loss: 0.0000
Step 32200 (epoch 41.18), Elapsed: 8891.0 ms, LR: 0.0000, Loss: 0.0187 Hint loss: 0.0000
Step 32300 (epoch 41.30), Elapsed: 8887.3 ms, LR: 0.0000, Loss: 0.0186 Hint loss: 0.0000
Step 32400 (epoch 41.43), Elapsed: 8884.8 ms, LR: 0.0000, Loss: 0.0185 Hint loss: 0.0000
Step 32500 (epoch 41.56), Elapsed: 8891.0 ms, LR: 0.0000, Loss: 0.0185 Hint loss: 0.0000
Step 32600 (epoch 41.69), Elapsed: 8899.9 ms, LR: 0.0000, Loss: 0.0187 Hint loss: 0.0000
Step 32700 (epoch 41.82), Elapsed: 8898.7 ms, LR: 0.0000, Loss: 0.0189 Hint loss: 0.0000
Step 32800 (epoch 41.94), Elapsed: 8895.0 ms, LR: 0.0000, Loss: 0.0185 Hint loss: 0.0000

[Validation] Epoch: 42.00, Elapsed: 2679.9 ms, Error: 1.20

Step 32900 (epoch 42.07), Elapsed: 8899.9 ms, LR: 0.0000, Loss: 0.0189 Hint loss: 0.0000
Step 33000 (epoch 42.20), Elapsed: 8888.5 ms, LR: 0.0000, Loss: 0.0193 Hint loss: 0.0000
Step 33100 (epoch 42.33), Elapsed: 8894.6 ms, LR: 0.0000, Loss: 0.0186 Hint loss: 0.0000
Step 33200 (epoch 42.46), Elapsed: 8888.2 ms, LR: 0.0000, Loss: 0.0188 Hint loss: 0.0000
Step 33300 (epoch 42.58), Elapsed: 8891.2 ms, LR: 0.0000, Loss: 0.0186 Hint loss: 0.0000
Step 33400 (epoch 42.71), Elapsed: 8892.3 ms, LR: 0.0000, Loss: 0.0194 Hint loss: 0.0000
Step 33500 (epoch 42.84), Elapsed: 8893.0 ms, LR: 0.0000, Loss: 0.0190 Hint loss: 0.0000
Step 33600 (epoch 42.97), Elapsed: 8904.3 ms, LR: 0.0000, Loss: 0.0183 Hint loss: 0.0000

[Validation] Epoch: 43.00, Elapsed: 2672.3 ms, Error: 1.18

Step 33700 (epoch 43.09), Elapsed: 8900.8 ms, LR: 0.0000, Loss: 0.0197 Hint loss: 0.0000
Step 33800 (epoch 43.22), Elapsed: 8894.1 ms, LR: 0.0000, Loss: 0.0193 Hint loss: 0.0000
Step 33900 (epoch 43.35), Elapsed: 8893.8 ms, LR: 0.0000, Loss: 0.0188 Hint loss: 0.0000
Step 34000 (epoch 43.48), Elapsed: 8893.8 ms, LR: 0.0000, Loss: 0.0182 Hint loss: 0.0000
Step 34100 (epoch 43.61), Elapsed: 8893.1 ms, LR: 0.0000, Loss: 0.0184 Hint loss: 0.0000
Step 34200 (epoch 43.73), Elapsed: 8897.0 ms, LR: 0.0000, Loss: 0.0181 Hint loss: 0.0000
Step 34300 (epoch 43.86), Elapsed: 8897.3 ms, LR: 0.0000, Loss: 0.0187 Hint loss: 0.0000
Step 34400 (epoch 43.99), Elapsed: 8896.5 ms, LR: 0.0000, Loss: 0.0204 Hint loss: 0.0000

[Validation] Epoch: 44.00, Elapsed: 2678.7 ms, Error: 1.18

Step 34500 (epoch 44.12), Elapsed: 8895.7 ms, LR: 0.0000, Loss: 0.0185 Hint loss: 0.0000
Step 34600 (epoch 44.25), Elapsed: 8897.4 ms, LR: 0.0000, Loss: 0.0184 Hint loss: 0.0000
Step 34700 (epoch 44.37), Elapsed: 8904.0 ms, LR: 0.0000, Loss: 0.0184 Hint loss: 0.0000
Step 34800 (epoch 44.50), Elapsed: 8897.3 ms, LR: 0.0000, Loss: 0.0187 Hint loss: 0.0000
Step 34900 (epoch 44.63), Elapsed: 8899.6 ms, LR: 0.0000, Loss: 0.0193 Hint loss: 0.0000
Step 35000 (epoch 44.76), Elapsed: 8900.3 ms, LR: 0.0000, Loss: 0.0181 Hint loss: 0.0000
Step 35100 (epoch 44.88), Elapsed: 8890.6 ms, LR: 0.0000, Loss: 0.0194 Hint loss: 0.0000

[Validation] Epoch: 45.00, Elapsed: 2682.0 ms, Error: 1.12

Step 35200 (epoch 45.01), Elapsed: 8897.9 ms, LR: 0.0000, Loss: 0.0180 Hint loss: 0.0000
Step 35300 (epoch 45.14), Elapsed: 8891.9 ms, LR: 0.0000, Loss: 0.0180 Hint loss: 0.0000
Step 35400 (epoch 45.27), Elapsed: 8894.4 ms, LR: 0.0000, Loss: 0.0186 Hint loss: 0.0000
Step 35500 (epoch 45.40), Elapsed: 8895.4 ms, LR: 0.0000, Loss: 0.0181 Hint loss: 0.0000
Step 35600 (epoch 45.52), Elapsed: 8895.9 ms, LR: 0.0000, Loss: 0.0180 Hint loss: 0.0000
Step 35700 (epoch 45.65), Elapsed: 8888.6 ms, LR: 0.0000, Loss: 0.0177 Hint loss: 0.0000
Step 35800 (epoch 45.78), Elapsed: 8895.7 ms, LR: 0.0000, Loss: 0.0194 Hint loss: 0.0000
Step 35900 (epoch 45.91), Elapsed: 8902.1 ms, LR: 0.0000, Loss: 0.0177 Hint loss: 0.0000

[Validation] Epoch: 46.00, Elapsed: 2671.5 ms, Error: 1.20

Step 36000 (epoch 46.04), Elapsed: 8889.2 ms, LR: 0.0000, Loss: 0.0183 Hint loss: 0.0000
Step 36100 (epoch 46.16), Elapsed: 8901.4 ms, LR: 0.0000, Loss: 0.0176 Hint loss: 0.0000
Step 36200 (epoch 46.29), Elapsed: 8900.5 ms, LR: 0.0000, Loss: 0.0181 Hint loss: 0.0000
Step 36300 (epoch 46.42), Elapsed: 8886.2 ms, LR: 0.0000, Loss: 0.0176 Hint loss: 0.0000
Step 36400 (epoch 46.55), Elapsed: 8895.5 ms, LR: 0.0000, Loss: 0.0177 Hint loss: 0.0000
Step 36500 (epoch 46.68), Elapsed: 8889.3 ms, LR: 0.0000, Loss: 0.0176 Hint loss: 0.0000
Step 36600 (epoch 46.80), Elapsed: 8902.1 ms, LR: 0.0000, Loss: 0.0194 Hint loss: 0.0000
Step 36700 (epoch 46.93), Elapsed: 8898.2 ms, LR: 0.0000, Loss: 0.0177 Hint loss: 0.0000

[Validation] Epoch: 47.00, Elapsed: 2669.2 ms, Error: 1.20

Step 36800 (epoch 47.06), Elapsed: 8887.5 ms, LR: 0.0000, Loss: 0.0178 Hint loss: 0.0000
Step 36900 (epoch 47.19), Elapsed: 8904.5 ms, LR: 0.0000, Loss: 0.0174 Hint loss: 0.0000
Step 37000 (epoch 47.31), Elapsed: 8892.0 ms, LR: 0.0000, Loss: 0.0177 Hint loss: 0.0000
Step 37100 (epoch 47.44), Elapsed: 8892.8 ms, LR: 0.0000, Loss: 0.0173 Hint loss: 0.0000
Step 37200 (epoch 47.57), Elapsed: 8895.4 ms, LR: 0.0000, Loss: 0.0173 Hint loss: 0.0000
Step 37300 (epoch 47.70), Elapsed: 8891.0 ms, LR: 0.0000, Loss: 0.0172 Hint loss: 0.0000
Step 37400 (epoch 47.83), Elapsed: 8892.1 ms, LR: 0.0000, Loss: 0.0173 Hint loss: 0.0000
Step 37500 (epoch 47.95), Elapsed: 8891.1 ms, LR: 0.0000, Loss: 0.0171 Hint loss: 0.0000

[Validation] Epoch: 48.00, Elapsed: 2671.5 ms, Error: 1.16

Step 37600 (epoch 48.08), Elapsed: 8891.9 ms, LR: 0.0000, Loss: 0.0172 Hint loss: 0.0000
Step 37700 (epoch 48.21), Elapsed: 8891.9 ms, LR: 0.0000, Loss: 0.0172 Hint loss: 0.0000
Step 37800 (epoch 48.34), Elapsed: 8894.7 ms, LR: 0.0000, Loss: 0.0181 Hint loss: 0.0000
Step 37900 (epoch 48.47), Elapsed: 8891.1 ms, LR: 0.0000, Loss: 0.0174 Hint loss: 0.0000
Step 38000 (epoch 48.59), Elapsed: 8906.5 ms, LR: 0.0000, Loss: 0.0170 Hint loss: 0.0000
Step 38100 (epoch 48.72), Elapsed: 8894.0 ms, LR: 0.0000, Loss: 0.0169 Hint loss: 0.0000
Step 38200 (epoch 48.85), Elapsed: 8889.7 ms, LR: 0.0000, Loss: 0.0168 Hint loss: 0.0000
Step 38300 (epoch 48.98), Elapsed: 8900.5 ms, LR: 0.0000, Loss: 0.0171 Hint loss: 0.0000

[Validation] Epoch: 49.00, Elapsed: 2671.5 ms, Error: 1.17

Step 38400 (epoch 49.10), Elapsed: 8895.1 ms, LR: 0.0000, Loss: 0.0172 Hint loss: 0.0000
Step 38500 (epoch 49.23), Elapsed: 8894.4 ms, LR: 0.0000, Loss: 0.0168 Hint loss: 0.0000
Step 38600 (epoch 49.36), Elapsed: 8897.7 ms, LR: 0.0000, Loss: 0.0168 Hint loss: 0.0000
Step 38700 (epoch 49.49), Elapsed: 8890.9 ms, LR: 0.0000, Loss: 0.0165 Hint loss: 0.0000
Step 38800 (epoch 49.62), Elapsed: 8898.0 ms, LR: 0.0000, Loss: 0.0165 Hint loss: 0.0000
Step 38900 (epoch 49.74), Elapsed: 8889.2 ms, LR: 0.0000, Loss: 0.0168 Hint loss: 0.0000
Step 39000 (epoch 49.87), Elapsed: 8890.9 ms, LR: 0.0000, Loss: 0.0171 Hint loss: 0.0000

[Validation] Epoch: 50.00, Elapsed: 2666.5 ms, Error: 1.23

Step 39100 (epoch 50.00), Elapsed: 8901.8 ms, LR: 0.0000, Loss: 0.0166 Hint loss: 0.0000
```

* DCN without hint objective
```
[Validation] Epoch: 40.00, Elapsed: 8977.9 ms, Error: 1.51

Step 31300 (epoch 40.03), Elapsed: 11141.9 ms, LR: 0.0000, Loss: 0.0353 Hint loss: 106.4182
Step 31400 (epoch 40.15), Elapsed: 11117.0 ms, LR: 0.0000, Loss: 0.0384 Hint loss: 99.3636
Step 31500 (epoch 40.28), Elapsed: 11160.9 ms, LR: 0.0000, Loss: 0.0373 Hint loss: 103.5755
Step 31600 (epoch 40.41), Elapsed: 11208.3 ms, LR: 0.0000, Loss: 0.0443 Hint loss: 100.2382
Step 31700 (epoch 40.54), Elapsed: 11348.3 ms, LR: 0.0000, Loss: 0.0385 Hint loss: 100.2246
Step 31800 (epoch 40.66), Elapsed: 11690.9 ms, LR: 0.0000, Loss: 0.0364 Hint loss: 104.1788
Step 31900 (epoch 40.79), Elapsed: 12064.7 ms, LR: 0.0000, Loss: 0.0350 Hint loss: 96.9056
Step 32000 (epoch 40.92), Elapsed: 12124.7 ms, LR: 0.0000, Loss: 0.0349 Hint loss: 98.5311

[Validation] Epoch: 41.00, Elapsed: 9539.6 ms, Error: 1.45

Step 32100 (epoch 41.05), Elapsed: 12038.4 ms, LR: 0.0000, Loss: 0.0348 Hint loss: 98.7128
Step 32200 (epoch 41.18), Elapsed: 12061.8 ms, LR: 0.0000, Loss: 0.0346 Hint loss: 105.0991
Step 32300 (epoch 41.30), Elapsed: 11972.6 ms, LR: 0.0000, Loss: 0.0391 Hint loss: 101.4495
Step 32400 (epoch 41.43), Elapsed: 11996.3 ms, LR: 0.0000, Loss: 0.0365 Hint loss: 97.0989
Step 32500 (epoch 41.56), Elapsed: 12026.7 ms, LR: 0.0000, Loss: 0.0354 Hint loss: 97.3752
Step 32600 (epoch 41.69), Elapsed: 12057.3 ms, LR: 0.0000, Loss: 0.0350 Hint loss: 97.3694
Step 32700 (epoch 41.82), Elapsed: 11985.4 ms, LR: 0.0000, Loss: 0.0347 Hint loss: 97.0989
Step 32800 (epoch 41.94), Elapsed: 12171.2 ms, LR: 0.0000, Loss: 0.0359 Hint loss: 99.2992

[Validation] Epoch: 42.00, Elapsed: 9553.1 ms, Error: 1.47

Step 32900 (epoch 42.07), Elapsed: 12059.4 ms, LR: 0.0000, Loss: 0.0359 Hint loss: 98.3476
Step 33000 (epoch 42.20), Elapsed: 11998.7 ms, LR: 0.0000, Loss: 0.0343 Hint loss: 97.3388
Step 33100 (epoch 42.33), Elapsed: 11974.4 ms, LR: 0.0000, Loss: 0.0461 Hint loss: 98.1271
Step 33200 (epoch 42.46), Elapsed: 11993.7 ms, LR: 0.0000, Loss: 0.0386 Hint loss: 99.2695
Step 33300 (epoch 42.58), Elapsed: 11950.6 ms, LR: 0.0000, Loss: 0.0452 Hint loss: 101.7823
Step 33400 (epoch 42.71), Elapsed: 11977.1 ms, LR: 0.0000, Loss: 0.0344 Hint loss: 94.3337
Step 33500 (epoch 42.84), Elapsed: 12014.9 ms, LR: 0.0000, Loss: 0.0357 Hint loss: 94.7903
Step 33600 (epoch 42.97), Elapsed: 12031.8 ms, LR: 0.0000, Loss: 0.0355 Hint loss: 98.9343

[Validation] Epoch: 43.00, Elapsed: 9492.7 ms, Error: 1.63

Step 33700 (epoch 43.09), Elapsed: 12035.9 ms, LR: 0.0000, Loss: 0.0370 Hint loss: 97.6984
Step 33800 (epoch 43.22), Elapsed: 12048.6 ms, LR: 0.0000, Loss: 0.0361 Hint loss: 102.7230
Step 33900 (epoch 43.35), Elapsed: 12023.8 ms, LR: 0.0000, Loss: 0.0383 Hint loss: 100.5490
Step 34000 (epoch 43.48), Elapsed: 11969.9 ms, LR: 0.0000, Loss: 0.0436 Hint loss: 93.8176
Step 34100 (epoch 43.61), Elapsed: 12019.8 ms, LR: 0.0000, Loss: 0.0355 Hint loss: 87.0270
Step 34200 (epoch 43.73), Elapsed: 12103.3 ms, LR: 0.0000, Loss: 0.0351 Hint loss: 87.2292
Step 34300 (epoch 43.86), Elapsed: 12113.0 ms, LR: 0.0000, Loss: 0.0352 Hint loss: 95.8945
Step 34400 (epoch 43.99), Elapsed: 12064.2 ms, LR: 0.0000, Loss: 0.0351 Hint loss: 90.4202

[Validation] Epoch: 44.00, Elapsed: 9571.9 ms, Error: 1.47

Step 34500 (epoch 44.12), Elapsed: 12053.7 ms, LR: 0.0000, Loss: 0.0383 Hint loss: 94.2170
Step 34600 (epoch 44.25), Elapsed: 12053.3 ms, LR: 0.0000, Loss: 0.0342 Hint loss: 97.6082
Step 34700 (epoch 44.37), Elapsed: 11996.5 ms, LR: 0.0000, Loss: 0.0359 Hint loss: 98.0818
Step 34800 (epoch 44.50), Elapsed: 12080.9 ms, LR: 0.0000, Loss: 0.0728 Hint loss: 106.9778
Step 34900 (epoch 44.63), Elapsed: 12072.5 ms, LR: 0.0000, Loss: 0.0350 Hint loss: 90.6349
Step 35000 (epoch 44.76), Elapsed: 11934.2 ms, LR: 0.0000, Loss: 0.0344 Hint loss: 97.5697
Step 35100 (epoch 44.88), Elapsed: 12060.2 ms, LR: 0.0000, Loss: 0.0342 Hint loss: 96.3839

[Validation] Epoch: 45.00, Elapsed: 9547.2 ms, Error: 1.53

Step 35200 (epoch 45.01), Elapsed: 11972.9 ms, LR: 0.0000, Loss: 0.0348 Hint loss: 95.7466
Step 35300 (epoch 45.14), Elapsed: 12013.2 ms, LR: 0.0000, Loss: 0.0350 Hint loss: 89.9611
Step 35400 (epoch 45.27), Elapsed: 11927.5 ms, LR: 0.0000, Loss: 0.0340 Hint loss: 96.7048
Step 35500 (epoch 45.40), Elapsed: 11973.7 ms, LR: 0.0000, Loss: 0.0390 Hint loss: 98.8128
Step 35600 (epoch 45.52), Elapsed: 11975.4 ms, LR: 0.0000, Loss: 0.0364 Hint loss: 92.7995
Step 35700 (epoch 45.65), Elapsed: 11968.5 ms, LR: 0.0000, Loss: 0.0347 Hint loss: 88.1528
Step 35800 (epoch 45.78), Elapsed: 12070.6 ms, LR: 0.0000, Loss: 0.0358 Hint loss: 97.7022
Step 35900 (epoch 45.91), Elapsed: 12127.7 ms, LR: 0.0000, Loss: 0.0388 Hint loss: 95.1412

[Validation] Epoch: 46.00, Elapsed: 9533.5 ms, Error: 1.57

Step 36000 (epoch 46.04), Elapsed: 11993.4 ms, LR: 0.0000, Loss: 0.0351 Hint loss: 90.7602
Step 36100 (epoch 46.16), Elapsed: 11940.5 ms, LR: 0.0000, Loss: 0.0336 Hint loss: 91.0333
Step 36200 (epoch 46.29), Elapsed: 11995.6 ms, LR: 0.0000, Loss: 0.0493 Hint loss: 98.7799
Step 36300 (epoch 46.42), Elapsed: 11896.7 ms, LR: 0.0000, Loss: 0.0346 Hint loss: 94.8729
Step 36400 (epoch 46.55), Elapsed: 11938.3 ms, LR: 0.0000, Loss: 0.0377 Hint loss: 93.6889
Step 36500 (epoch 46.68), Elapsed: 11905.0 ms, LR: 0.0000, Loss: 0.0340 Hint loss: 97.5995
Step 36600 (epoch 46.80), Elapsed: 11932.9 ms, LR: 0.0000, Loss: 0.0356 Hint loss: 89.2434
Step 36700 (epoch 46.93), Elapsed: 12095.2 ms, LR: 0.0000, Loss: 0.0339 Hint loss: 96.6894

[Validation] Epoch: 47.00, Elapsed: 9524.3 ms, Error: 1.44

Step 36800 (epoch 47.06), Elapsed: 11907.0 ms, LR: 0.0000, Loss: 0.0334 Hint loss: 93.1803
Step 36900 (epoch 47.19), Elapsed: 11888.5 ms, LR: 0.0000, Loss: 0.0339 Hint loss: 97.3097
Step 37000 (epoch 47.31), Elapsed: 12019.8 ms, LR: 0.0000, Loss: 0.0339 Hint loss: 93.5960
Step 37100 (epoch 47.44), Elapsed: 11947.5 ms, LR: 0.0000, Loss: 0.0345 Hint loss: 95.7494
Step 37200 (epoch 47.57), Elapsed: 12018.3 ms, LR: 0.0000, Loss: 0.0343 Hint loss: 99.7749
Step 37300 (epoch 47.70), Elapsed: 11917.0 ms, LR: 0.0000, Loss: 0.0337 Hint loss: 96.0459
Step 37400 (epoch 47.83), Elapsed: 12005.8 ms, LR: 0.0000, Loss: 0.0334 Hint loss: 94.5175
Step 37500 (epoch 47.95), Elapsed: 12047.4 ms, LR: 0.0000, Loss: 0.0344 Hint loss: 99.3840

[Validation] Epoch: 48.00, Elapsed: 9538.6 ms, Error: 1.52

Step 37600 (epoch 48.08), Elapsed: 11984.2 ms, LR: 0.0000, Loss: 0.0355 Hint loss: 100.0203
Step 37700 (epoch 48.21), Elapsed: 11918.2 ms, LR: 0.0000, Loss: 0.0354 Hint loss: 99.8090
Step 37800 (epoch 48.34), Elapsed: 11924.8 ms, LR: 0.0000, Loss: 0.0435 Hint loss: 92.5132
Step 37900 (epoch 48.47), Elapsed: 12035.7 ms, LR: 0.0000, Loss: 0.0340 Hint loss: 104.9256
Step 38000 (epoch 48.59), Elapsed: 11964.8 ms, LR: 0.0000, Loss: 0.0375 Hint loss: 97.8294
Step 38100 (epoch 48.72), Elapsed: 11927.5 ms, LR: 0.0000, Loss: 0.0358 Hint loss: 98.1366
Step 38200 (epoch 48.85), Elapsed: 12040.7 ms, LR: 0.0000, Loss: 0.0350 Hint loss: 100.5713
Step 38300 (epoch 48.98), Elapsed: 12048.5 ms, LR: 0.0000, Loss: 0.0371 Hint loss: 96.9352

[Validation] Epoch: 49.00, Elapsed: 9550.7 ms, Error: 1.49

Step 38400 (epoch 49.10), Elapsed: 11953.9 ms, LR: 0.0000, Loss: 0.0333 Hint loss: 93.2281
Step 38500 (epoch 49.23), Elapsed: 11978.9 ms, LR: 0.0000, Loss: 0.0338 Hint loss: 95.0153
Step 38600 (epoch 49.36), Elapsed: 11991.4 ms, LR: 0.0000, Loss: 0.0335 Hint loss: 93.3222
Step 38700 (epoch 49.49), Elapsed: 11861.3 ms, LR: 0.0000, Loss: 0.0341 Hint loss: 95.0970
Step 38800 (epoch 49.62), Elapsed: 11906.8 ms, LR: 0.0000, Loss: 0.0330 Hint loss: 92.9684
Step 38900 (epoch 49.74), Elapsed: 11963.8 ms, LR: 0.0000, Loss: 0.0336 Hint loss: 99.8751
Step 39000 (epoch 49.87), Elapsed: 12048.9 ms, LR: 0.0000, Loss: 0.0427 Hint loss: 94.3697

[Validation] Epoch: 50.00, Elapsed: 9590.4 ms, Error: 1.52

Step 39100 (epoch 50.00), Elapsed: 11963.6 ms, LR: 0.0000, Loss: 0.0350 Hint loss: 99.4407
```

* DCN with hint objective
```
[Validation] Epoch: 40.00, Elapsed: 9609.4 ms, Error: 1.55

Step 31300 (epoch 40.03), Elapsed: 12123.1 ms, LR: 0.0000, Loss: 0.0481 Hint loss: 8.1279
Step 31400 (epoch 40.15), Elapsed: 12236.9 ms, LR: 0.0000, Loss: 0.0514 Hint loss: 8.3041
Step 31500 (epoch 40.28), Elapsed: 12204.1 ms, LR: 0.0000, Loss: 0.0324 Hint loss: 7.9792
Step 31600 (epoch 40.41), Elapsed: 12058.2 ms, LR: 0.0000, Loss: 0.0317 Hint loss: 8.2941
Step 31700 (epoch 40.54), Elapsed: 11958.0 ms, LR: 0.0000, Loss: 0.0323 Hint loss: 7.7475
Step 31800 (epoch 40.66), Elapsed: 11999.3 ms, LR: 0.0000, Loss: 0.0331 Hint loss: 8.3960
Step 31900 (epoch 40.79), Elapsed: 12032.4 ms, LR: 0.0000, Loss: 0.0316 Hint loss: 7.9445
Step 32000 (epoch 40.92), Elapsed: 12034.6 ms, LR: 0.0000, Loss: 0.0334 Hint loss: 7.7186

[Validation] Epoch: 41.00, Elapsed: 9630.6 ms, Error: 1.46

Step 32100 (epoch 41.05), Elapsed: 12037.9 ms, LR: 0.0000, Loss: 0.0323 Hint loss: 8.1867
Step 32200 (epoch 41.18), Elapsed: 12221.0 ms, LR: 0.0000, Loss: 0.0336 Hint loss: 8.1771
Step 32300 (epoch 41.30), Elapsed: 12109.5 ms, LR: 0.0000, Loss: 0.0317 Hint loss: 7.8697
Step 32400 (epoch 41.43), Elapsed: 12017.5 ms, LR: 0.0000, Loss: 0.0329 Hint loss: 7.9616
Step 32500 (epoch 41.56), Elapsed: 11979.5 ms, LR: 0.0000, Loss: 0.0322 Hint loss: 8.0880
Step 32600 (epoch 41.69), Elapsed: 11979.8 ms, LR: 0.0000, Loss: 0.0320 Hint loss: 7.9887
Step 32700 (epoch 41.82), Elapsed: 11951.3 ms, LR: 0.0000, Loss: 0.0333 Hint loss: 7.8962
Step 32800 (epoch 41.94), Elapsed: 12028.6 ms, LR: 0.0000, Loss: 0.0377 Hint loss: 8.3715

[Validation] Epoch: 42.00, Elapsed: 9580.7 ms, Error: 1.38

Step 32900 (epoch 42.07), Elapsed: 12022.4 ms, LR: 0.0000, Loss: 0.0318 Hint loss: 7.9486
Step 33000 (epoch 42.20), Elapsed: 12077.8 ms, LR: 0.0000, Loss: 0.0333 Hint loss: 8.0088
Step 33100 (epoch 42.33), Elapsed: 12083.2 ms, LR: 0.0000, Loss: 0.0337 Hint loss: 7.9642
Step 33200 (epoch 42.46), Elapsed: 12052.0 ms, LR: 0.0000, Loss: 0.0315 Hint loss: 7.5406
Step 33300 (epoch 42.58), Elapsed: 12023.2 ms, LR: 0.0000, Loss: 0.0314 Hint loss: 7.7493
Step 33400 (epoch 42.71), Elapsed: 11962.2 ms, LR: 0.0000, Loss: 0.0340 Hint loss: 8.0980
Step 33500 (epoch 42.84), Elapsed: 12017.4 ms, LR: 0.0000, Loss: 0.0325 Hint loss: 8.1223
Step 33600 (epoch 42.97), Elapsed: 12108.6 ms, LR: 0.0000, Loss: 0.0325 Hint loss: 7.9907

[Validation] Epoch: 43.00, Elapsed: 9537.5 ms, Error: 1.49

Step 33700 (epoch 43.09), Elapsed: 12111.6 ms, LR: 0.0000, Loss: 0.0359 Hint loss: 8.1180
Step 33800 (epoch 43.22), Elapsed: 12073.1 ms, LR: 0.0000, Loss: 0.0506 Hint loss: 8.0158
Step 33900 (epoch 43.35), Elapsed: 11993.8 ms, LR: 0.0000, Loss: 0.0460 Hint loss: 7.6198
Step 34000 (epoch 43.48), Elapsed: 12035.3 ms, LR: 0.0000, Loss: 0.0326 Hint loss: 7.9989
Step 34100 (epoch 43.61), Elapsed: 12006.9 ms, LR: 0.0000, Loss: 0.0310 Hint loss: 7.9017
Step 34200 (epoch 43.73), Elapsed: 12107.2 ms, LR: 0.0000, Loss: 0.0311 Hint loss: 8.0928
Step 34300 (epoch 43.86), Elapsed: 12037.1 ms, LR: 0.0000, Loss: 0.0311 Hint loss: 7.9462
Step 34400 (epoch 43.99), Elapsed: 11923.5 ms, LR: 0.0000, Loss: 0.0310 Hint loss: 7.9574

[Validation] Epoch: 44.00, Elapsed: 9496.2 ms, Error: 1.43

Step 34500 (epoch 44.12), Elapsed: 11975.0 ms, LR: 0.0000, Loss: 0.0335 Hint loss: 7.9774
Step 34600 (epoch 44.25), Elapsed: 11998.1 ms, LR: 0.0000, Loss: 0.0320 Hint loss: 7.9656
Step 34700 (epoch 44.37), Elapsed: 11930.8 ms, LR: 0.0000, Loss: 0.0308 Hint loss: 7.8411
Step 34800 (epoch 44.50), Elapsed: 11908.0 ms, LR: 0.0000, Loss: 0.0318 Hint loss: 8.0539
Step 34900 (epoch 44.63), Elapsed: 12002.1 ms, LR: 0.0000, Loss: 0.0324 Hint loss: 8.0651
Step 35000 (epoch 44.76), Elapsed: 11955.4 ms, LR: 0.0000, Loss: 0.0327 Hint loss: 7.8720
Step 35100 (epoch 44.88), Elapsed: 12025.4 ms, LR: 0.0000, Loss: 0.0310 Hint loss: 8.1035

[Validation] Epoch: 45.00, Elapsed: 9493.5 ms, Error: 1.38

Step 35200 (epoch 45.01), Elapsed: 12045.9 ms, LR: 0.0000, Loss: 0.0313 Hint loss: 7.9145
Step 35300 (epoch 45.14), Elapsed: 12017.4 ms, LR: 0.0000, Loss: 0.0346 Hint loss: 7.9799
Step 35400 (epoch 45.27), Elapsed: 12000.1 ms, LR: 0.0000, Loss: 0.0310 Hint loss: 8.1268
Step 35500 (epoch 45.40), Elapsed: 11978.7 ms, LR: 0.0000, Loss: 0.0321 Hint loss: 8.0472
Step 35600 (epoch 45.52), Elapsed: 11945.5 ms, LR: 0.0000, Loss: 0.0341 Hint loss: 7.9871
Step 35700 (epoch 45.65), Elapsed: 11901.3 ms, LR: 0.0000, Loss: 0.0307 Hint loss: 8.0926
Step 35800 (epoch 45.78), Elapsed: 11939.7 ms, LR: 0.0000, Loss: 0.0307 Hint loss: 7.6016
Step 35900 (epoch 45.91), Elapsed: 11911.8 ms, LR: 0.0000, Loss: 0.0314 Hint loss: 8.0678

[Validation] Epoch: 46.00, Elapsed: 9385.3 ms, Error: 1.36

Step 36000 (epoch 46.04), Elapsed: 11906.9 ms, LR: 0.0000, Loss: 0.0345 Hint loss: 7.8840
Step 36100 (epoch 46.16), Elapsed: 11968.3 ms, LR: 0.0000, Loss: 0.0316 Hint loss: 7.9791
Step 36200 (epoch 46.29), Elapsed: 11940.8 ms, LR: 0.0000, Loss: 0.0312 Hint loss: 7.6835
Step 36300 (epoch 46.42), Elapsed: 11862.3 ms, LR: 0.0000, Loss: 0.0324 Hint loss: 7.9401
Step 36400 (epoch 46.55), Elapsed: 11878.9 ms, LR: 0.0000, Loss: 0.0309 Hint loss: 7.8467
Step 36500 (epoch 46.68), Elapsed: 11906.3 ms, LR: 0.0000, Loss: 0.0323 Hint loss: 7.8245
Step 36600 (epoch 46.80), Elapsed: 11861.8 ms, LR: 0.0000, Loss: 0.0642 Hint loss: 8.1247
Step 36700 (epoch 46.93), Elapsed: 11920.2 ms, LR: 0.0000, Loss: 0.0334 Hint loss: 8.1486

[Validation] Epoch: 47.00, Elapsed: 9412.6 ms, Error: 1.42

Step 36800 (epoch 47.06), Elapsed: 11894.1 ms, LR: 0.0000, Loss: 0.0355 Hint loss: 7.7621
Step 36900 (epoch 47.19), Elapsed: 11969.3 ms, LR: 0.0000, Loss: 0.0319 Hint loss: 7.9595
Step 37000 (epoch 47.31), Elapsed: 11946.7 ms, LR: 0.0000, Loss: 0.0305 Hint loss: 7.8896
Step 37100 (epoch 47.44), Elapsed: 11904.3 ms, LR: 0.0000, Loss: 0.0328 Hint loss: 7.8951
Step 37200 (epoch 47.57), Elapsed: 11906.0 ms, LR: 0.0000, Loss: 0.0356 Hint loss: 8.0274
Step 37300 (epoch 47.70), Elapsed: 11949.6 ms, LR: 0.0000, Loss: 0.0312 Hint loss: 7.7374
Step 37400 (epoch 47.83), Elapsed: 11866.1 ms, LR: 0.0000, Loss: 0.0328 Hint loss: 8.0280
Step 37500 (epoch 47.95), Elapsed: 11903.3 ms, LR: 0.0000, Loss: 0.0310 Hint loss: 7.9039

[Validation] Epoch: 48.00, Elapsed: 9437.3 ms, Error: 1.44

Step 37600 (epoch 48.08), Elapsed: 11926.6 ms, LR: 0.0000, Loss: 0.0309 Hint loss: 8.1272
Step 37700 (epoch 48.21), Elapsed: 11935.0 ms, LR: 0.0000, Loss: 0.0406 Hint loss: 7.7479
Step 37800 (epoch 48.34), Elapsed: 11912.1 ms, LR: 0.0000, Loss: 0.0309 Hint loss: 7.9133
Step 37900 (epoch 48.47), Elapsed: 11905.8 ms, LR: 0.0000, Loss: 0.0333 Hint loss: 7.9182
Step 38000 (epoch 48.59), Elapsed: 11941.3 ms, LR: 0.0000, Loss: 0.0327 Hint loss: 7.8363
Step 38100 (epoch 48.72), Elapsed: 11861.4 ms, LR: 0.0000, Loss: 0.0317 Hint loss: 8.0948
Step 38200 (epoch 48.85), Elapsed: 11954.8 ms, LR: 0.0000, Loss: 0.0305 Hint loss: 8.1091
Step 38300 (epoch 48.98), Elapsed: 11930.6 ms, LR: 0.0000, Loss: 0.0301 Hint loss: 7.7706

[Validation] Epoch: 49.00, Elapsed: 9410.3 ms, Error: 1.37

Step 38400 (epoch 49.10), Elapsed: 11838.3 ms, LR: 0.0000, Loss: 0.0302 Hint loss: 8.0595
Step 38500 (epoch 49.23), Elapsed: 11963.3 ms, LR: 0.0000, Loss: 0.0300 Hint loss: 7.8085
Step 38600 (epoch 49.36), Elapsed: 10211.8 ms, LR: 0.0000, Loss: 0.0302 Hint loss: 8.0124
Step 38700 (epoch 49.49), Elapsed: 10077.2 ms, LR: 0.0000, Loss: 0.0300 Hint loss: 8.0691
Step 38800 (epoch 49.62), Elapsed: 10666.8 ms, LR: 0.0000, Loss: 0.0299 Hint loss: 8.0896
Step 38900 (epoch 49.74), Elapsed: 10513.9 ms, LR: 0.0000, Loss: 0.0309 Hint loss: 7.8712
Step 39000 (epoch 49.87), Elapsed: 10479.1 ms, LR: 0.0000, Loss: 0.0299 Hint loss: 7.8189

[Validation] Epoch: 50.00, Elapsed: 8526.5 ms, Error: 1.31

Step 39100 (epoch 50.00), Elapsed: 10857.9 ms, LR: 0.0000, Loss: 0.0300 Hint loss: 7.7300
```

