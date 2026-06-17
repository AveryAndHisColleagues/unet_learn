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

# Final results
| Exp | Model    | Loss        | Aug | Attention | Best IoU |
| --- | -------- | ----------- | --- | --------- | -------: |
| E1  | MiniUNet | BCE+Dice    | ✗   | ✗         |    0.511 |
| E2  | ResUNet  | BCE+Dice    | ✗   | ✗         |    0.312 |
| E3  | ResUNet  | BCE+Tversky | ✗   | ✗         |    0.557 |
| E4  | MiniUNet | BCE+Tversky | ✗   | ✗         |    0.486 |
| E5  | ResUNet  | BCE+Tversky | ✓   | ✗         |    0.590 |
| E6  | ResUNet  | BCE+Tversky | ✓   | SE        |    0.594 |
| E7  | ResUNet  | BCE+Tversky | ✓   | CBAM      |    0.594 |


## Takeaways

- ResUNet + BCE+Tversky achieves the best validation IoU.
- ResUNet + BCE+Dice performs poorly, showing that this architecture is loss-sensitive.
- MiniUNet + BCE+Dice remains a strong baseline.

## Chatters
This juvenile project aims to help me understand the process of image segmentation. I didn't expect this project would go on for too long. But working on this ablation experiment gives me a glimpse of how research and innovation should be conducted and structured(though what I'm doing rn is far from any of them). I need more time to go over this project. Hope that this is the beginning of something. 

Just learned Messi's hat-trick against Algeria. Vamos Argentina carajo. I'm gonna wear my Argentina jersey when playing football later.
