# Mini U-Net on DRIVE

Structure:
- DoubleConv
- Down
- Up
- Skip Connection

Loss:
- BCE + Dice

Metric:
- IoU

Best Val IoU:
0.49

# DRIVE Ablation Summary

| Experiment         |   Best Val IoU |   Best Epoch |   Final Train IoU |   Final Val IoU |
|:-------------------|---------------:|-------------:|------------------:|----------------:|
| MiniUNet_BCEDice   |         0.5110 |           49 |            0.6863 |          0.4369 |
| ResUNet_BCEDice    |         0.3122 |           49 |            0.4094 |          0.2927 |
| ResUNet_BCETversky |         0.5567 |           39 |            0.8002 |          0.5543 |

## Takeaways

- ResUNet + BCE+Tversky achieves the best validation IoU.
- ResUNet + BCE+Dice performs poorly, showing that this architecture is loss-sensitive.
- MiniUNet + BCE+Dice remains a strong baseline.