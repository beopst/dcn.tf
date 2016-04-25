# Dynamic Capacity Networks using Tensorflow

Dynamic Capacity Networks (DCN; http://arxiv.org/abs/1511.07838) implementation using Tensorflow. DCN reduces the number of computations by applying high-capacity networks to some selected input patches. 

## Dataset

* [Cluttered MNIST](https://github.com/deepmind/mnist-cluttered)

## NOTES

* Computation time issue

Although DCN can greatly reduce the number of computations, it may consume more computation time since it requires sequantial processes (extract input patches of interest, and then feed them into high-capacity networks). Computation time can be reduced if the input patches are concatenated for convolution.

## Logs
* Coarse Model
```
[Validation] Epoch: 97.00, Elapsed: 713.6 ms, Error: 2.24

Step 75900 (epoch 97.06), Elapsed: 2586.6 ms, LR: 0.0000, Loss: 0.4766 Hint loss: 0.0000
Step 76000 (epoch 97.19), Elapsed: 2591.0 ms, LR: 0.0000, Loss: 0.3880 Hint loss: 0.0000
Step 76100 (epoch 97.31), Elapsed: 2589.4 ms, LR: 0.0000, Loss: 0.4077 Hint loss: 0.0000
Step 76200 (epoch 97.44), Elapsed: 2585.3 ms, LR: 0.0000, Loss: 0.3968 Hint loss: 0.0000
Step 76300 (epoch 97.57), Elapsed: 2597.1 ms, LR: 0.0000, Loss: 0.3753 Hint loss: 0.0000
Step 76400 (epoch 97.70), Elapsed: 2591.6 ms, LR: 0.0000, Loss: 0.4137 Hint loss: 0.0000
Step 76500 (epoch 97.83), Elapsed: 2596.5 ms, LR: 0.0000, Loss: 0.4332 Hint loss: 0.0000
Step 76600 (epoch 97.95), Elapsed: 2580.4 ms, LR: 0.0000, Loss: 0.3913 Hint loss: 0.0000

[Validation] Epoch: 98.00, Elapsed: 734.8 ms, Error: 2.29

Step 76700 (epoch 98.08), Elapsed: 2589.7 ms, LR: 0.0000, Loss: 0.3718 Hint loss: 0.0000
Step 76800 (epoch 98.21), Elapsed: 2587.6 ms, LR: 0.0000, Loss: 0.4405 Hint loss: 0.0000
Step 76900 (epoch 98.34), Elapsed: 2588.1 ms, LR: 0.0000, Loss: 0.4405 Hint loss: 0.0000
Step 77000 (epoch 98.47), Elapsed: 2588.0 ms, LR: 0.0000, Loss: 0.4438 Hint loss: 0.0000
Step 77100 (epoch 98.59), Elapsed: 2595.9 ms, LR: 0.0000, Loss: 0.4575 Hint loss: 0.0000
Step 77200 (epoch 98.72), Elapsed: 2593.8 ms, LR: 0.0000, Loss: 0.4271 Hint loss: 0.0000
Step 77300 (epoch 98.85), Elapsed: 2584.6 ms, LR: 0.0000, Loss: 0.3985 Hint loss: 0.0000
Step 77400 (epoch 98.98), Elapsed: 2591.0 ms, LR: 0.0000, Loss: 0.4216 Hint loss: 0.0000

[Validation] Epoch: 99.00, Elapsed: 705.9 ms, Error: 2.31

Step 77500 (epoch 99.10), Elapsed: 2599.0 ms, LR: 0.0000, Loss: 0.4258 Hint loss: 0.0000
Step 77600 (epoch 99.23), Elapsed: 2631.0 ms, LR: 0.0000, Loss: 0.3995 Hint loss: 0.0000
Step 77700 (epoch 99.36), Elapsed: 2588.9 ms, LR: 0.0000, Loss: 0.4273 Hint loss: 0.0000
Step 77800 (epoch 99.49), Elapsed: 2601.1 ms, LR: 0.0000, Loss: 0.3912 Hint loss: 0.0000
Step 77900 (epoch 99.62), Elapsed: 2594.1 ms, LR: 0.0000, Loss: 0.4042 Hint loss: 0.0000
Step 78000 (epoch 99.74), Elapsed: 2577.6 ms, LR: 0.0000, Loss: 0.4457 Hint loss: 0.0000
Step 78100 (epoch 99.87), Elapsed: 2583.2 ms, LR: 0.0000, Loss: 0.3910 Hint loss: 0.0000

[Validation] Epoch: 100.00, Elapsed: 717.6 ms, Error: 2.27
```

* Fine Model
```
[Validation] Epoch: 97.00, Elapsed: 2664.2 ms, Error: 1.05

Step 75900 (epoch 97.06), Elapsed: 9383.1 ms, LR: 0.0000, Loss: 0.4106 Hint loss: 0.0000
Step 76000 (epoch 97.19), Elapsed: 9386.8 ms, LR: 0.0000, Loss: 0.3062 Hint loss: 0.0000
Step 76100 (epoch 97.31), Elapsed: 9375.5 ms, LR: 0.0000, Loss: 0.3575 Hint loss: 0.0000
Step 76200 (epoch 97.44), Elapsed: 9368.4 ms, LR: 0.0000, Loss: 0.3406 Hint loss: 0.0000
Step 76300 (epoch 97.57), Elapsed: 9390.9 ms, LR: 0.0000, Loss: 0.3770 Hint loss: 0.0000
Step 76400 (epoch 97.70), Elapsed: 9400.5 ms, LR: 0.0000, Loss: 0.3359 Hint loss: 0.0000
Step 76500 (epoch 97.83), Elapsed: 9385.3 ms, LR: 0.0000, Loss: 0.3764 Hint loss: 0.0000
Step 76600 (epoch 97.95), Elapsed: 9391.5 ms, LR: 0.0000, Loss: 0.3284 Hint loss: 0.0000

[Validation] Epoch: 98.00, Elapsed: 2661.3 ms, Error: 1.07

Step 76700 (epoch 98.08), Elapsed: 9396.5 ms, LR: 0.0000, Loss: 0.3622 Hint loss: 0.0000
Step 76800 (epoch 98.21), Elapsed: 9395.2 ms, LR: 0.0000, Loss: 0.3446 Hint loss: 0.0000
Step 76900 (epoch 98.34), Elapsed: 9383.6 ms, LR: 0.0000, Loss: 0.3392 Hint loss: 0.0000
Step 77000 (epoch 98.47), Elapsed: 9383.7 ms, LR: 0.0000, Loss: 0.3118 Hint loss: 0.0000
Step 77100 (epoch 98.59), Elapsed: 9381.5 ms, LR: 0.0000, Loss: 0.3294 Hint loss: 0.0000
Step 77200 (epoch 98.72), Elapsed: 9384.0 ms, LR: 0.0000, Loss: 0.3574 Hint loss: 0.0000
Step 77300 (epoch 98.85), Elapsed: 9376.7 ms, LR: 0.0000, Loss: 0.3096 Hint loss: 0.0000
Step 77400 (epoch 98.98), Elapsed: 9379.8 ms, LR: 0.0000, Loss: 0.3261 Hint loss: 0.0000

[Validation] Epoch: 99.00, Elapsed: 2667.1 ms, Error: 1.05

Step 77500 (epoch 99.10), Elapsed: 9387.7 ms, LR: 0.0000, Loss: 0.3622 Hint loss: 0.0000
Step 77600 (epoch 99.23), Elapsed: 9375.2 ms, LR: 0.0000, Loss: 0.3153 Hint loss: 0.0000
Step 77700 (epoch 99.36), Elapsed: 9390.4 ms, LR: 0.0000, Loss: 0.3809 Hint loss: 0.0000
Step 77800 (epoch 99.49), Elapsed: 9383.8 ms, LR: 0.0000, Loss: 0.3542 Hint loss: 0.0000
Step 77900 (epoch 99.62), Elapsed: 9384.3 ms, LR: 0.0000, Loss: 0.3765 Hint loss: 0.0000
Step 78000 (epoch 99.74), Elapsed: 9399.7 ms, LR: 0.0000, Loss: 0.3441 Hint loss: 0.0000
Step 78100 (epoch 99.87), Elapsed: 9401.5 ms, LR: 0.0000, Loss: 0.3542 Hint loss: 0.0000

[Validation] Epoch: 100.00, Elapsed: 2676.4 ms, Error: 1.07
```
