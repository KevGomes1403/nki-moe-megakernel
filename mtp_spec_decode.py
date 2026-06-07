"""Self-speculative decoding for Qwen3.6-A3B.

Per round: draft x_{t+2} with the 1-layer MTP head, verify the 2-token block
[x_{t+1}, x_{t+2}] with the backbone, accept the draft iff it matches the
backbone's argmax, then commit the accepted DeltaNet/conv state. Greedy output
is identical to plain decode (accepted tokens are the backbone's own argmax).

Requires a build with the draft + verify graphs:
``build_inference_config(..., mtp_spec_decode=True)``. All device plumbing lives
on the model (spec_prefill / draft / verify / commit_specdec / read_loop_state /
reset_spec_state); this module is just the host loop.
"""

import torch


def _num_dn_layers(config):
    return sum(t == "linear_attention" for t in config.layer_types)


def spec_decode(model, tokenizer, config, prompt, max_new_tokens):
    """Greedy spec-decode one prompt. Returns (token_ids incl. prompt, accept_rate)."""
    dtype = config.neuron_config.torch_dtype
    num_dn = _num_dn_layers(config)
    seq_len = config.neuron_config.seq_len
    eos = tokenizer.eos_token_id
    seq_ids = torch.tensor([0], dtype=torch.int32)
    zeros_block = torch.zeros((1, 2, config.hidden_size), dtype=dtype)

    def i32(v):
        return torch.tensor(v, dtype=torch.int32)

    prompt_ids = tokenizer(prompt).input_ids
    p = len(prompt_ids)

    model.reset_spec_state()
    x_next, h = model.spec_prefill(prompt_ids)   # x_p, h_{p-1}
    t = p - 1                                     # index of the last committed token

    tokens = list(prompt_ids)
    accepts = rounds = 0

    # t carries the last committed position; x_next is the token at t+1 and h_t
    # predicts it. Stop on EOS or when the 2-token block would exceed seq_len.
    while len(tokens) - p < max_new_tokens and t + 3 < seq_len:
        rounds += 1
        dl = model.draft(h, i32([[x_next]]), i32([[t + 1]]), seq_ids)
        x_draft = int(dl.float().argmax(-1).flatten()[0])

        vlog = model.verify(
            zeros_block, i32([[x_next, x_draft]]), i32([[t + 1, t + 2]]), seq_ids
        )[0].float()  # [1, 2, vocab]
        true_t2 = int(vlog[:, 0].argmax(-1).flatten()[0])
        accept = x_draft == true_t2

        model.commit_specdec(1 if accept else 0, num_dn)  # 1 -> S2, 0 -> S1
        trunk = model.read_loop_state("verify_trunk_buffer")  # [1, 2, H]

        if accept:
            tokens += [x_next, x_draft]
            # keep the draft head's KV contiguous over the committed positions
            model.draft(trunk[:, 0:1].to(dtype), i32([[x_draft]]), i32([[t + 2]]), seq_ids)
            x_next = int(vlog[:, 1].argmax(-1).flatten()[0])  # x_{t+3}
            h = trunk[:, 1:2].to(dtype)                       # h_{t+2}
            t += 2
            accepts += 1
        else:
            tokens.append(x_next)
            x_next = true_t2                                  # corrected x_{t+2}
            h = trunk[:, 0:1].to(dtype)                       # h_{t+1}
            t += 1

        if x_next == eos:
            tokens.append(x_next)
            break

    return tokens, (accepts / rounds if rounds else 0.0)
