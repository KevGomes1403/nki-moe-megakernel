import torch


def qwen3_moe_rms_norm(
    hidden_states,
    weight,
):
    input_dtype = hidden_states.dtype
    hidden_states_float32 = hidden_states.to(torch.float32)
    variance = hidden_states_float32.pow(2).mean(-1, keepdim=True)
    hidden_states_float32_times_rsqrt_variance = hidden_states_float32 * torch.rsqrt(variance + 1e-06)
    return weight * hidden_states_float32_times_rsqrt_variance.to(input_dtype)


def silu_activation(input):
    return torch.nn.functional.silu(input)


def qwen3_moe_top_k_router(
    hidden_states,
    weight,
):
    hidden_states_reshaped = hidden_states.reshape(-1, 2048)
    router_logits = torch.nn.functional.linear(hidden_states_reshaped, weight)  # (seq_len, num_experts)
    softmax_router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(softmax_router_logits, 8, dim=-1)  # (seq_len, 8)
    router_top_value_normed = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_scores = router_top_value_normed.to(softmax_router_logits.dtype)
    return softmax_router_logits, router_scores, router_indices


def qwen3_moe_rotary_embedding(
    x,
    position_ids,
    inv_freq,
):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * 1.0
    sin = emb.sin() * 1.0

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    unsqueeze_dim=1
):
    cos_unsqueezed = cos.unsqueeze(unsqueeze_dim)
    sin_unsqueezed = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_unsqueezed) + (rotate_half(q) * sin_unsqueezed)
    k_embed = (k * cos_unsqueezed) + (rotate_half(k) * sin_unsqueezed)
    return q_embed, k_embed


def repeat_kv(x, n_rep):
    batch, n_kv_heads, seq_len, head_dim = x.shape
    # Insert a new dimension for repetition
    # Repeat along the new dimension
    # Merge the head and repetition dimensions
    return x[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


def sdpa_attention_forward(
    query,
    key,
    value,
    attention_mask,
):
    key = repeat_kv(key, 8)
    value = repeat_kv(value, 8)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=0.08838834764831845,
        is_causal=True
    )
    attn_output_transposed = attn_output.transpose(1, 2).contiguous()

    return attn_output_transposed


def qwen3_moe_attention(
    hidden_states,
    cos,
    sin,
    attention_mask,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 128)

    query_states = qwen3_moe_rms_norm(
        torch.nn.functional.linear(
            hidden_states,
            q_proj_weight,
        ).view(hidden_shape),
        q_norm_weight,
    ).transpose(1, 2)
    
    key_states = qwen3_moe_rms_norm(
        torch.nn.functional.linear(
            hidden_states,
            k_proj_weight,
        ).view(hidden_shape),
        k_norm_weight,
    ).transpose(1, 2)
    
    value_states = torch.nn.functional.linear(
        hidden_states,
        v_proj_weight,
    ).view(hidden_shape).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin
    )

    attn_output = sdpa_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
    )

    attn_output_reshaped = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output_reshaped_projected = torch.nn.functional.linear(
        attn_output_reshaped,
        o_proj_weight,
    )

    return attn_output_reshaped_projected


def qwen3_moe_experts(
    hidden_states,
    top_k_index,
    top_k_weights,
    # In the model code, there are two single parameters:
    # - gate_up_proj
    # - down_proj
    # However, in the state dict, there are three lists of parameters:
    # - *.gate_proj.weight
    # - *.up_proj.weight
    # - *.down_proj.weight
    # We align ourselves with the state dict.
    gate_proj_weight_list,
    up_proj_weight_list,
    down_proj_weight_list,
):
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=128)
        expert_mask_permuted = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask_permuted.sum(dim=(-1, -2)), 0).nonzero()

    final_hidden_states = torch.zeros_like(hidden_states)
    for expert_idx_tensor in expert_hit:
        expert_idx = expert_idx_tensor[0]
        if expert_idx == 128:
            continue

        top_k_pos, token_idx = torch.where(expert_mask_permuted[expert_idx])
        
        gate = torch.nn.functional.linear(
            hidden_states[token_idx],
            gate_proj_weight_list[expert_idx]
        )

        up = torch.nn.functional.linear(
            hidden_states[token_idx],
            up_proj_weight_list[expert_idx]
        )
        
        current_hidden_states = torch.nn.functional.linear(
            silu_activation(gate) * up,
            down_proj_weight_list[expert_idx]
        ) * top_k_weights[token_idx, top_k_pos, None]
        
        final_hidden_states.index_add_(
            0,
            token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


def qwen3_moe_sparse_moe_block(
    hidden_states,
    # gate.weight
    gate_weight,
    # experts.*.gate_proj.weight
    experts_gate_proj_weight_list,
    # experts.*.up_proj.weight
    experts_up_proj_weight_list,
    # experts.*.down_proj.weight
    experts_down_proj_weight_list,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    
    _, routing_weights, selected_experts = qwen3_moe_top_k_router(
        hidden_states_reshaped,
        gate_weight,
    )
    
    final_hidden_states = qwen3_moe_experts(
        hidden_states_reshaped,
        selected_experts,
        routing_weights,
        experts_gate_proj_weight_list,
        experts_up_proj_weight_list,
        experts_down_proj_weight_list,
    )
    
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def qwen3_moe_decoder_layer(
    hidden_states,
    attention_mask,
    cos,
    sin,
    # input_layernorm.weight
    input_layernorm_weight,
    # self_attn.q_proj.weight
    self_attn_q_proj_weight,
    # self_attn.k_proj.weight
    self_attn_k_proj_weight,
    # self_attn.v_proj.weight
    self_attn_v_proj_weight,
    # self_attn.o_proj.weight
    self_attn_o_proj_weight,
    # self_attn.q_norm.weight
    self_attn_q_norm_weight,
    # self_attn.k_norm.weight
    self_attn_k_norm_weight,
    # post_attention_layernorm.weight
    post_attention_layernorm_weight,
    # mlp.gate.weight
    mlp_gate_weight,
    # mlp.experts.*.gate_proj.weight
    mlp_experts_gate_proj_weight_list,
    # mlp.experts.*.up_proj.weight
    mlp_experts_up_proj_weight_list,
    # mlp.experts.*.down_proj.weight
    mlp_experts_down_proj_weight_list,
):
    # Self Attention
    self_attn_residual = hidden_states
    
    normed_hidden_states = qwen3_moe_rms_norm(
        hidden_states,
        input_layernorm_weight,
    )
    
    self_attn_output = qwen3_moe_attention(
        normed_hidden_states,
        cos,
        sin,
        attention_mask,
        self_attn_q_proj_weight,
        self_attn_k_proj_weight,
        self_attn_v_proj_weight,
        self_attn_o_proj_weight,
        self_attn_q_norm_weight,
        self_attn_k_norm_weight,
    )
    
    self_attn_output_with_residual = self_attn_residual + self_attn_output

    # Fully Connected
    mlp_residual = self_attn_output_with_residual
    
    normed_self_attn_output_with_residual = qwen3_moe_rms_norm(
        self_attn_output_with_residual,
        post_attention_layernorm_weight
    )
    
    mlp_output = qwen3_moe_sparse_moe_block(
        normed_self_attn_output_with_residual,
        mlp_gate_weight,
        mlp_experts_gate_proj_weight_list,
        mlp_experts_up_proj_weight_list,
        mlp_experts_down_proj_weight_list,
    )
    
    mlp_output_with_residual = mlp_residual + mlp_output
    
    return mlp_output_with_residual


def qwen3_moe_model(
    input_ids,
    attention_mask,
    position_ids,
    # embed_tokens.weight
    embed_tokens_weight,
    # rotary_emb.inv_freq
    rotary_emb_inv_freq,
    # layers.*.input_layernorm.weight
    layers_input_layernorm_weight_list,
    # layers.*.self_attn.q_proj.weight
    layers_self_attn_q_proj_weight_list,
    # layers.*.self_attn.k_proj.weight
    layers_self_attn_k_proj_weight_list,
    # layers.*.self_attn.v_proj.weight
    layers_self_attn_v_proj_weight_list,
    # layers.*.self_attn.o_proj.weight
    layers_self_attn_o_proj_weight_list,
    # layers.*.self_attn.q_norm.weight
    layers_self_attn_q_norm_weight_list,
    # layers.*.self_attn.k_norm.weight
    layers_self_attn_k_norm_weight_list,
    # layers.*.post_attention_layernorm.weight
    layers_post_attention_layernorm_weight_list,
    # layers.*.mlp.gate.weight
    layers_mlp_gate_weight_list,
    # layers.*.mlp.experts.*.gate_proj.weight
    layers_mlp_experts_gate_proj_weight_list_list,
    # layers.*.mlp.experts.*.up_proj.weight
    layers_mlp_experts_up_proj_weight_list_list,
    # layers.*.mlp.experts.*.down_proj.weight
    layers_mlp_experts_down_proj_weight_list_list,
    # norm.weight
    norm_weight,
):
    batch_size, length = input_ids.shape
    device = input_ids.device

    # 4D boolean mask (B, 1, L, L)
    causal_mask = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))[None,None,:,:].expand(batch_size, 1, length, length)
    attention_mask_expanded = attention_mask[:,None,None,:].to(dtype=torch.bool)
    final_mask = causal_mask & attention_mask_expanded

    hidden_states = torch.nn.functional.embedding(
        input_ids,
        embed_tokens_weight
    )

    cos, sin = qwen3_moe_rotary_embedding(
        x=hidden_states,
        position_ids=position_ids,
        inv_freq=rotary_emb_inv_freq
    )

    for (
        input_layernorm_weight,
        self_attn_q_proj_weight,
        self_attn_k_proj_weight,
        self_attn_v_proj_weight,
        self_attn_o_proj_weight,
        self_attn_q_norm_weight,
        self_attn_k_norm_weight,
        post_attention_layernorm_weight,
        mlp_gate_weight,
        mlp_experts_gate_proj_weight_list,
        mlp_experts_up_proj_weight_list,
        mlp_experts_down_proj_weight_list,
    ) in zip(
        layers_input_layernorm_weight_list,
        layers_self_attn_q_proj_weight_list,
        layers_self_attn_k_proj_weight_list,
        layers_self_attn_v_proj_weight_list,
        layers_self_attn_o_proj_weight_list,
        layers_self_attn_q_norm_weight_list,
        layers_self_attn_k_norm_weight_list,
        layers_post_attention_layernorm_weight_list,
        layers_mlp_gate_weight_list,
        layers_mlp_experts_gate_proj_weight_list_list,
        layers_mlp_experts_up_proj_weight_list_list,
        layers_mlp_experts_down_proj_weight_list_list,
    ):
        hidden_states = qwen3_moe_decoder_layer(
            hidden_states,
            final_mask,
            cos,
            sin,
            input_layernorm_weight,
            self_attn_q_proj_weight,
            self_attn_k_proj_weight,
            self_attn_v_proj_weight,
            self_attn_o_proj_weight,
            self_attn_q_norm_weight,
            self_attn_k_norm_weight,
            post_attention_layernorm_weight,
            mlp_gate_weight,
            mlp_experts_gate_proj_weight_list,
            mlp_experts_up_proj_weight_list,
            mlp_experts_down_proj_weight_list,
        )
    
    normed_hidden_states = qwen3_moe_rms_norm(
        hidden_states=hidden_states,
        weight=norm_weight,
    )

    return normed_hidden_states
    

def qwen3_moe_for_causal_lm(
    input_ids,
    attention_mask,
    position_ids,
    # model.embed_tokens.weight
    model_embed_tokens_weight,
    # model.rotary_emb.inv_freq
    model_rotary_emb_inv_freq,
    # model.layers.*.input_layernorm.weight
    model_layers_input_layernorm_weight_list,
    # model.layers.*.self_attn.q_proj.weight
    model_layers_self_attn_q_proj_weight_list,
    # model_layers.*.self_attn.k_proj.weight
    model_layers_self_attn_k_proj_weight_list,
    # model_layers.*.self_attn.v_proj.weight
    model_layers_self_attn_v_proj_weight_list,
    # model.layers.*.self_attn.o_proj.weight
    model_layers_self_attn_o_proj_weight_list,
    # model.layers.*.self_attn.q_norm.weight
    model_layers_self_attn_q_norm_weight_list,
    # model.layers.*.self_attn.k_norm.weight
    model_layers_self_attn_k_norm_weight_list,
    # model.layers.*.post_attention_layernorm.weight
    model_layers_post_attention_layernorm_weight_list,
    # model.layers.*.mlp.gate.weight
    model_layers_mlp_gate_weight_list,
    # model.layers.*.mlp.experts.*.gate_proj.weight
    model_layers_mlp_experts_gate_proj_weight_list_list,
    # model.layers.*.mlp.experts.*.up_proj.weight
    model_layers_mlp_experts_up_proj_weight_list_list,
    # model.layers.*.mlp.experts.*.down_proj.weight
    model_layers_mlp_experts_down_proj_weight_list_list,
    # model.norm.weight
    model_norm_weight,
    # lm_head.weight
    lm_head_weight,
):
    hidden_states = qwen3_moe_model(
        input_ids,
        attention_mask,
        position_ids,
        model_embed_tokens_weight,
        model_rotary_emb_inv_freq,
        model_layers_input_layernorm_weight_list,
        model_layers_self_attn_q_proj_weight_list,
        model_layers_self_attn_k_proj_weight_list,
        model_layers_self_attn_v_proj_weight_list,
        model_layers_self_attn_o_proj_weight_list,
        model_layers_self_attn_q_norm_weight_list,
        model_layers_self_attn_k_norm_weight_list,
        model_layers_post_attention_layernorm_weight_list,
        model_layers_mlp_gate_weight_list,
        model_layers_mlp_experts_gate_proj_weight_list_list,
        model_layers_mlp_experts_up_proj_weight_list_list,
        model_layers_mlp_experts_down_proj_weight_list_list,
        model_norm_weight,
    )

    logits = torch.nn.functional.linear(
        hidden_states,
        lm_head_weight,
    )

    return logits


if __name__ == '__main__':
    from glob import glob
    from os.path import join
    from safetensors_layer_grabber import yield_keys_and_tensors
    from sys import stderr

    parameters = {}
    parameters.setdefault('model', {}).setdefault('rotary_emb', {})['inv_freq'] = torch.Tensor(
        [
            1.0000e+00, 8.0584e-01, 6.4938e-01, 5.2330e-01, 4.2170e-01, 3.3982e-01, 2.7384e-01, 2.2067e-01,
            1.7783e-01, 1.4330e-01, 1.1548e-01, 9.3057e-02, 7.4989e-02, 6.0430e-02, 4.8697e-02, 3.9242e-02,
            3.1623e-02, 2.5483e-02, 2.0535e-02, 1.6548e-02, 1.3335e-02, 1.0746e-02, 8.6596e-03, 6.9783e-03,
            5.6234e-03, 4.5316e-03, 3.6517e-03, 2.9427e-03, 2.3714e-03, 1.9110e-03, 1.5399e-03, 1.2409e-03,
            1.0000e-03, 8.0584e-04, 6.4938e-04, 5.2330e-04, 4.2170e-04, 3.3982e-04, 2.7384e-04, 2.2067e-04,
            1.7783e-04, 1.4330e-04, 1.1548e-04, 9.3057e-05, 7.4989e-05, 6.0430e-05, 4.8697e-05, 3.9242e-05,
            3.1623e-05, 2.5483e-05, 2.0535e-05, 1.6548e-05, 1.3335e-05, 1.0746e-05, 8.6596e-06, 6.9783e-06,
            5.6234e-06, 4.5316e-06, 3.6517e-06, 2.9427e-06, 2.3714e-06, 1.9110e-06, 1.5399e-06, 1.2409e-06,
        ]
    )

    safetensors_file_names = glob(join('Qwen', 'Qwen3-30B-A3B', '*.safetensors'))
    for tensor_key, tensor in yield_keys_and_tensors(safetensors_file_names):
        tensor_key_components = tensor_key.split('.')
        if not tensor_key_components:
            print('Invalid tensor key:', tensor_key)
            continue

        prefixes = tensor_key_components[:-1]
        last_tensor_key_component = tensor_key_components[-1]
        
        current_level = parameters
        for prefix in prefixes:
            current_level = current_level.setdefault(prefix, {})

        current_level[last_tensor_key_component] = tensor

    # inputs_and_outputs = torch.load('model.layers.31.mlp.pt', weights_only=False)
    # output = qwen3_moe_sparse_moe_block             (
    #     inputs_and_outputs['hidden_states'],
    #     parameters['model']['layers']['31']['mlp']['gate']['weight'],
    #     [parameters['model']['layers']['31']['mlp']['experts'][str(j)]['gate_proj']['weight'] for j in range(128)],
    #     [parameters['model']['layers']['31']['mlp']['experts'][str(j)]['up_proj']['weight'] for j in range(128)],
    #     [parameters['model']['layers']['31']['mlp']['experts'][str(j)]['down_proj']['weight'] for j in range(128)],
    # )
    # print(output)
    # print(inputs_and_outputs['return'])
    
    # inputs_and_outputs = torch.load('model.pt', weights_only=False)
    # output = qwen3_moe_model(
    #     inputs_and_outputs['input_ids'],
    #     inputs_and_outputs['attention_mask'],
    #     inputs_and_outputs['position_ids'],
    #     parameters['model']['embed_tokens']['weight'],
    #     parameters['model']['rotary_emb']['inv_freq'],
    #     [parameters['model']['layers'][str(i)]['input_layernorm']['weight'] for i in range(48)],
    #     [parameters['model']['layers'][str(i)]['self_attn']['q_proj']['weight'] for i in range(48)],
    #     [parameters['model']['layers'][str(i)]['self_attn']['k_proj']['weight'] for i in range(48)],
    #     [parameters['model']['layers'][str(i)]['self_attn']['v_proj']['weight'] for i in range(48)],
    #     [parameters['model']['layers'][str(i)]['self_attn']['o_proj']['weight'] for i in range(48)],
    #     [parameters['model']['layers'][str(i)]['self_attn']['q_norm']['weight'] for i in range(48)],
    #     [parameters['model']['layers'][str(i)]['self_attn']['k_norm']['weight'] for i in range(48)],
    #     [parameters['model']['layers'][str(i)]['post_attention_layernorm']['weight'] for i in range(48)],
    #     [parameters['model']['layers'][str(i)]['mlp']['gate']['weight'] for i in range(48)],
    #     [[parameters['model']['layers'][str(i)]['mlp']['experts'][str(j)]['gate_proj']['weight'] for j in range(128)] for i in range(48)],
    #     [[parameters['model']['layers'][str(i)]['mlp']['experts'][str(j)]['up_proj']['weight'] for j in range(128)] for i in range(48)],
    #     [[parameters['model']['layers'][str(i)]['mlp']['experts'][str(j)]['down_proj']['weight'] for j in range(128)] for i in range(48)],
    #     parameters['model']['norm']['weight'],
    # )
    # print(output)
    # print(inputs_and_outputs['return'])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-30B-A3B"
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(['Give me a short introduction to large language model.'])
    print(inputs)

    input_ids = torch.LongTensor(inputs['input_ids'])
    attention_mask = torch.BoolTensor(inputs['attention_mask'])

    from autoregressive_language_model_generate import autoregressive_language_model_generate

    def model(input_ids, attention_mask, position_ids):
        return qwen3_moe_for_causal_lm(
            input_ids,
            attention_mask,
            position_ids,
            parameters['model']['embed_tokens']['weight'],
            parameters['model']['rotary_emb']['inv_freq'],
            [parameters['model']['layers'][str(i)]['input_layernorm']['weight'] for i in range(48)],
            [parameters['model']['layers'][str(i)]['self_attn']['q_proj']['weight'] for i in range(48)],
            [parameters['model']['layers'][str(i)]['self_attn']['k_proj']['weight'] for i in range(48)],
            [parameters['model']['layers'][str(i)]['self_attn']['v_proj']['weight'] for i in range(48)],
            [parameters['model']['layers'][str(i)]['self_attn']['o_proj']['weight'] for i in range(48)],
            [parameters['model']['layers'][str(i)]['self_attn']['q_norm']['weight'] for i in range(48)],
            [parameters['model']['layers'][str(i)]['self_attn']['k_norm']['weight'] for i in range(48)],
            [parameters['model']['layers'][str(i)]['post_attention_layernorm']['weight'] for i in range(48)],
            [parameters['model']['layers'][str(i)]['mlp']['gate']['weight'] for i in range(48)],
            [[parameters['model']['layers'][str(i)]['mlp']['experts'][str(j)]['gate_proj']['weight'] for j in range(128)] for i in range(48)],
            [[parameters['model']['layers'][str(i)]['mlp']['experts'][str(j)]['up_proj']['weight'] for j in range(128)] for i in range(48)],
            [[parameters['model']['layers'][str(i)]['mlp']['experts'][str(j)]['down_proj']['weight'] for j in range(128)] for i in range(48)],
            parameters['model']['norm']['weight'],
            parameters['lm_head']['weight'],
        )

    gen = autoregressive_language_model_generate(
        model,
        input_ids,
        attention_mask
    )
    
    logits = next(gen)
    while True:
        # Implement your sampling logic here
        next_token_logits = logits[:, -1, :]
        top_k = 50
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_scores = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        
        # `next_tokens` has shape `(batch_size,)`
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        print(tokenizer.decode(next_tokens))
        
        # Send `next_tokens` to generator, receive `logits`
        logits = gen.send(next_tokens)