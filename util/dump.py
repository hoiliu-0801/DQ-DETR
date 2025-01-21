import torch
import numpy as np
import os

conv_features, enc_attn_weights, dec_attn_weights = [], [], []
ccm, channelgate, spatialgate, cgfe = [], [], [], []
def register_hooks(dqdetr_model):
    dqdetr_model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    )
    dqdetr_model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output)
    )
    dqdetr_model.transformer.CCM.ccm.register_forward_hook(
        lambda self, input, output: ccm.append(output)
    )
    dqdetr_model.transformer.CGFE.ChannelGate.register_forward_hook(
        lambda self, input, output: channelgate.append(output)
    )
    dqdetr_model.transformer.CGFE.SpatialGate.register_forward_hook(
        lambda self, input, output: spatialgate.append(output)
    )
    dqdetr_model.transformer.CGFE.register_forward_hook(
        lambda self, input, output: cgfe.append(output)
    )
    dqdetr_model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output)
    )

@torch.no_grad()
def save_data(targets, output):
    os.makedirs(output, exist_ok=True)
    for i, v in targets[0].items():
        np.save(os.path.join(output, f'target_{i}'), v.cpu().numpy())
    for k, v in conv_features[0].items():
        np.save(os.path.join(output, f'backbone_{k}'), v.tensors.cpu().numpy())
    np.save(os.path.join(output, 'enc'), enc_attn_weights[0].cpu().numpy())
    np.save(os.path.join(output, 'ccm'), ccm[0].cpu().numpy())
    for i, v in enumerate(channelgate):
        np.save(os.path.join(output, f'channelgate_{i}'), v.cpu().numpy())
    for i, v in enumerate(spatialgate):
        np.save(os.path.join(output, f'spatialgate_{i}'), v.cpu().numpy())
    np.save(os.path.join(output, 'cgfe'), cgfe[0].cpu().numpy())
    np.save(os.path.join(output, 'dec'), dec_attn_weights[0].cpu().numpy())
    print(f"Data saved to {output}")