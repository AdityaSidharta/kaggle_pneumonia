@snl I still have the same error. Here are the following steps that I perform to replicate this error

```
import torchvision
densenet = torchvision.models.densenet121(pretrained=True)
densenet.cuda()
```

Error => RuntimeError: CUDA error: all CUDA-capable devices are busy or unavailable