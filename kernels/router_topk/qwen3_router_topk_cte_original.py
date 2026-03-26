"""Original baseline kernel — x: [T, H]"""
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import core_barrier

H = 2048; E = 128; K = 8; P = 128; NUM_H_TILES = H // P

@nki.jit(platform_target="trn2")
def qwen3_router_topk_cte_original(
    x, w, router_logits, expert_affinities, expert_index,
):
    T = x.shape[0]
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    T_local = T // n_prgs
    T_offset = prg_id * T_local

    w_reshape = w.reshape((P, NUM_H_TILES, E))

    w_tiles = []
    x_tiles = []
    for ht in nl.affine_range(NUM_H_TILES):
        w_tile = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=w_tile,
            src=w_reshape.ap([[NUM_H_TILES * E, P], [1, E]], offset=ht * E),
        )
        w_tiles.append(w_tile)

    for ht in nl.affine_range(NUM_H_TILES):
        x_tile = nl.ndarray((P, T_local), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=x_tile,
            src=x.ap([[NUM_H_TILES, P], [H, T_local]], offset=T_offset * H + ht),
        )
        x_tiles.append(x_tile)

    router_logits_psum = nl.zeros((T_local, E), dtype=nl.float32, buffer=nl.psum)
    for ht in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(dst=router_logits_psum, stationary=x_tiles[ht], moving=w_tiles[ht])

    router_logits_sb = nl.ndarray((T_local, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=router_logits_sb, src=router_logits_psum)
    nisa.dma_copy(dst=router_logits.ap([[E, T_local], [1, E]], offset=T_offset * E), src=router_logits_sb)

    affinities_sb = nl.ndarray((T_local, E), dtype=nl.float32, buffer=nl.sbuf)
    negmax_sb = nl.ndarray((T_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=negmax_sb, op=nl.maximum, data=router_logits_sb, axis=1, negate=True, keepdims=True)
    inv_sum_sb = nl.ndarray((T_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=affinities_sb, op=nl.exp, data=router_logits_sb, bias=negmax_sb, reduce_op=nl.add, reduce_res=inv_sum_sb)
    nisa.reciprocal(dst=inv_sum_sb, data=inv_sum_sb)
    nisa.tensor_scalar(dst=affinities_sb, data=affinities_sb, op0=nl.multiply, operand0=inv_sum_sb)

    topk_vals_sb = nl.ndarray((T_local, K), dtype=nl.float32, buffer=nl.sbuf)
    topk_idx_sb = nl.ndarray((T_local, K), dtype=nl.uint32, buffer=nl.sbuf)
    top8_buf = nl.ndarray((T_local, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_buf, src=affinities_sb)
    nisa.tensor_copy(dst=topk_vals_sb, src=top8_buf[:, :K])
    idx8_buf = nl.ndarray((T_local, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=idx8_buf, data=affinities_sb, vals=top8_buf)
    nisa.tensor_copy(dst=topk_idx_sb, src=idx8_buf[:, :K])
    topk_idx_fp32_sb = nl.ndarray((T_local, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=topk_idx_fp32_sb, src=topk_idx_sb)
    nisa.dma_copy(dst=expert_index.ap([[K, T_local], [1, K]], offset=T_offset * K), src=topk_idx_sb)

    sum_topk_sb = nl.ndarray((T_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_topk_sb, op=nl.add, data=topk_vals_sb, axis=1, keepdims=True)
    nisa.reciprocal(dst=sum_topk_sb, data=sum_topk_sb)
    topk_vals_norm_sb = nl.ndarray((T_local, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=topk_vals_norm_sb, data=topk_vals_sb, op0=nl.multiply, operand0=sum_topk_sb)

    mask_sb = nl.ndarray((T_local, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mask_sb, value=0.0)
    expert_iota = nl.ndarray((P, E), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=expert_iota, pattern=[[1, E]], offset=0, channel_multiplier=0)
    check_buf = nl.ndarray((T_local, E), dtype=nl.float32, buffer=nl.sbuf)
    for k_slot in range(K):
        nisa.tensor_scalar(dst=check_buf[:T_local, :], op0=nl.equal, data=expert_iota[:T_local, :], operand0=topk_idx_fp32_sb[:T_local, k_slot])
        nisa.tensor_tensor(dst=mask_sb[:T_local, :], data1=mask_sb[:T_local, :], op=nl.add, data2=check_buf[:T_local, :])

    nisa.tensor_scalar(dst=affinities_sb, data=affinities_sb, op0=nl.multiply, operand0=sum_topk_sb)
    scattered_sb = nl.ndarray((T_local, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scattered_sb, data1=mask_sb, op=nl.multiply, data2=affinities_sb)
    nisa.dma_copy(dst=expert_affinities.ap([[E, T_local], [1, E]], offset=T_offset * E), src=scattered_sb)
    core_barrier(expert_affinities, cores=[0, 1])
    return router_logits, expert_affinities, expert_index
