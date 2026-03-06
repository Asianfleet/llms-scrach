"""
GPT-2 权重加载
"""
import numpy as np
import torch


def load_weights_into_gpt(gpt, params):
    """
    将预训练权重加载到 GPT-2 模型
    Args:
        gpt: GPT2 模型实例
        params: 预训练参数字典
    """

    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_q.weight = assign(
            gpt.transformer_blocks[b].attn.W_q.weight, q_w.T)
        gpt.transformer_blocks[b].attn.W_k.weight = assign(
            gpt.transformer_blocks[b].attn.W_k.weight, k_w.T)
        gpt.transformer_blocks[b].attn.W_v.weight = assign(
            gpt.transformer_blocks[b].attn.W_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        # qkv_bias=False 时 W_q/W_k/W_v 不包含 bias，需要跳过
        if gpt.transformer_blocks[b].attn.W_q.bias is not None:
            gpt.transformer_blocks[b].attn.W_q.bias = assign(
                gpt.transformer_blocks[b].attn.W_q.bias, q_b)
        if gpt.transformer_blocks[b].attn.W_k.bias is not None:
            gpt.transformer_blocks[b].attn.W_k.bias = assign(
                gpt.transformer_blocks[b].attn.W_k.bias, k_b)
        if gpt.transformer_blocks[b].attn.W_v.bias is not None:
            gpt.transformer_blocks[b].attn.W_v.bias = assign(
                gpt.transformer_blocks[b].attn.W_v.bias, v_b)

        gpt.transformer_blocks[b].attn.out_proj.weight = assign(
            gpt.transformer_blocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].attn.out_proj.bias = assign(
            gpt.transformer_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].ffn.layers[0].weight = assign(
            gpt.transformer_blocks[b].ffn.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].ffn.layers[0].bias = assign(
            gpt.transformer_blocks[b].ffn.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].ffn.layers[2].weight = assign(
            gpt.transformer_blocks[b].ffn.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].ffn.layers[2].bias = assign(
            gpt.transformer_blocks[b].ffn.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].layer_norm1.scale = assign(
            gpt.transformer_blocks[b].layer_norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].layer_norm1.shift = assign(
            gpt.transformer_blocks[b].layer_norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].layer_norm2.scale = assign(
            gpt.transformer_blocks[b].layer_norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].layer_norm2.shift = assign(
            gpt.transformer_blocks[b].layer_norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
