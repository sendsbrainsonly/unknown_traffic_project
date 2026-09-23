# TrafficFormer architecture

`word_pos_seg embedding -> 12-layer, 12-head Transformer (hidden 768, FFN 3072, GELU, dropout 0.1) -> first-token z_t (768) -> Linear(768,768)+tanh -> class head`. Historical training used NLL/softmax classification and official pretrained initialization. Stage17 strict pilot uses the identical architecture and random normal initialization because the official pretraining corpus exposure is undisclosed.
