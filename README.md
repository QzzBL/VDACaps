# VDACaps

Zhizhe Qian, Jing Mu, Feng Tian. "Ventral-Dorsal Attention Capsule Network for Facial Expression

Recognition". 

## Requirements
- Python >= 3.7
- PyTorch >= 1.7
- torchvision >= 0.9.1

## Training

- Step 1: download basic emotions dataset of CK+, and make sure it have the structure like following:

```
./CK+/
        
        anger/
            ***.png
            ...
            ***.png
        contempt/
            ***.png
            ...
            ***.png
        disgust/
        ...
        surprise

[Note] 0: anger; 1: contempt; 2: disgust; 3: fear; 4: happy; 5: sadness; 6: surprise
```

- Step 2: change ***data_path*** in *train.py* to your path 

- Step 3: run ```python train.py ```

## Note
Our experiment did not use the pre-trained model.
