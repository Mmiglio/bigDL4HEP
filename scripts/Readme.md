## Train
Launch the train with `launchTrain.sh`. 
It is possible to visualize loss/throughput by starting a tensorboard pointing to `logDir` 

```
tensorboard --logdir=logDir
```

Then you will find the tensorboard at `http://localhost:6006`.
Use `launchInference.sh` to plot the ROC curve to compare different models.