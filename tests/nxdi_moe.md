# HLO
1. Configuration (from neuron_config.json) 
                                                                                                                                                               
  ┌──────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────┐   
  │                           Key                            │                                            Value                                            │ 
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ Model                                                    │ Qwen3‑30B‑A3B (num_experts=128, num_experts_per_tok=8, hidden_size=2048,                    │ 
  │                                                          │ moe_intermediate_size=768, num_hidden_layers=48)                                            │ 
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ Storage dtype                                            │ bfloat16 (all weights), overrides_torch_dtype=True                                          │   
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ Parallelism                                              │ tp_degree=4, ep_degree=1, sp=False, lnc=2, world_size=4; TKG bucket 128                     │   
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤ 
  │ Per‑TP shard                                             │ I_TP = 768/4 = 192 intermediate units per rank                                              │   
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ Router                                                   │ act_fn="softmax", configured dtype=fp32 but weight is stored bf16 (compiled tensor in HLO   │   
  │                                                          │ is bf16), softmax executed in fp32                                                          │   
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ normalize_top_k_affinities                               │ True → F.normalize(x, p=1, dim=1, eps=1e-12)                                                │ 
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ moe_fused_nki_kernel_enabled                             │ None (false) → NOT the TKG fused kernel. Token generation goes through                      │   
  │                                                          │ MoE._forward_compute_bound → ExpertMLPsV2.forward_selective_loading                         │ 
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ glu_type="glu", hidden_act="silu",                       │                                                                                             │
  │ hidden_act_scaling_factor=1.0, hidden_act_bias=0.0, no   │                                                                                             │   
  │ gate/up clamp                                            │                                                                                             │
  ├──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤   
  │                                                          │ -O1 --enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none,      │
  │ Compiler flags                                           │ tensorizer: --enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2,                    │   
  │                                                          │ vector_dynamic_offsets DGE                                                                  │
  └──────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────┘   
                                                                  
  Consequence: every dot instruction in the HLO has precision_config=[DEFAULT, DEFAULT], but because of --enable-mixed-precision-accumulation all bf16 matmul  
  PSUMs are FP32 internally with the final store cast to the stated output dtype.
                                                                                                                                                               
  2. MoE dispatch path                                            

  For token generation (seq_len == 1, T=1, top_k=8):                                                                                                           
  - MoE.forward is called with residual=None → _forward_compute_bound.
  - rmsnorm on the MoE module is None (because init_tkg_module=False). Post‑attention RMSNorm is therefore applied outside the MoE module by                   
  NeuronQwen3MoeDecoderLayer.                                                                                                               
  - sequence_parallel_enabled=False so no scatter / gather on hidden states.                                                                                   
  - ep_enabled=False and token_shuffle_group_size=1.                        
  - In ExpertMLPsV2.forward: perc_experts_loaded = T·top_k / E = 8/128 = 0.0625 < DEFAULT_SELECTIVE_LOADING_THRESHOLD, so the runtime calls                    
  forward_selective_loading.                                                                                                                                   
                                                                                                                                                               
  The HLO op_names confirm this: …/MoE[.49][1]/_forward_compute_bound/ExpertMLPsV2[.49][1]/forward_selective_loading/….                                        
                                                                                                                                                               
  3. Tensor inputs/outputs of the MoE block                       
                                                                                                                                                               
  - Input (to mlp(...)): bf16[1,1,2048] = post‑attention hidden states (already normalized).                                                                   
  - Output (from mlp(...)): bf16[1,1,2048], identical on all 4 TP ranks after the all‑reduce.
  - Residual add (hidden_states = residual + mlp_output) happens in the decoder wrapper, not in the kernel.                                                    
                                                                                                                                                               
  The post_attention_layernorm that feeds the MoE:                                                                                                             
  %325 = bf16[1,1,2048] add(%17, attn_output)                    # post-attn residual                                                                          
  %326 = f32  convert(%325)                                      # bf16 → fp32                                                                                 
  %336 = f32  AwsNeuronRmsNorm(%326, weight_bf16, eps=1e-6_f32)  # backend_config="2" (dim=-1 on 2048)                                                         
  %337 = bf16 convert(%336)                                      # fp32 → bf16 → MoE input                                                                     
                                                                                                                                                               
  4. Shard‑local expert weights (per TP rank, bf16)                                                                                                            
                                                                                                                                                               
  Identified by parameter shape and layout from convert_qwen3_moe_hf_to_neuron_state_dict:                                                                     
                                                                                                                                                               
  ┌───────────────────────────────────────────────┬──────────────────────┬──────────────────────────────────────────────────────────────────────┐              
  │                     Param                     │      HLO shape       │                                Layout                                │
  ├───────────────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤              
  │ router.linear_router.weight (%319)            │ bf16[128, 2048]      │ [E, H], replicated across TP ranks                                   │
  ├───────────────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ expert_mlps.mlp_op.gate_up_proj.weight (%448) │ bf16[128, 2048, 384] │ [E, H, 2·I_TP]: gate in positions [0:192], up in positions [192:384] │              
  ├───────────────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤              
  │ expert_mlps.mlp_op.down_proj.weight (%432)    │ bf16[128, 192, 2048] │ [E, I_TP, H]                                                         │              
  └───────────────────────────────────────────────┴──────────────────────┴──────────────────────────────────────────────────────────────────────┘              
                                                                  
  5. Exact per‑op sequence (with precisions)                                                                                                                   
                                                                  
  All HLO instruction ids below are layer 0. Later layers repeat the same structure.                                                                           
                                                                  
  5a. Router projection — bf16 matmul                                                                                                                          
                                                                  
  %338 = bf16[1,2048] reshape(%337)                                  # hidden states (post-LN)                                                                 
  %320 = bf16[2048,128] transpose(%319)                              # router weight Wᵣᵀ                                                                       
  %339 = bf16[1,128] dot(%338, %320)                                 # router_logits; fp32 PSUM, bf16 store                                                    
                                                                     # contract: lhs=[1], rhs=[0]                                                              
  %341 = bf16[1,128] reshape(%339)                                   # flat (T=1, E=128)                                                                       
  - No bias.                                                                                                                                                   
  - Same on every TP rank (weight is replicated), so router_logits are identical across ranks — this guarantees consistent top‑k and affinities globally.      
                                                                                                                                                               
  5b. Top‑K — custom call on bf16 logits                                                                                                                       
                                                                                                                                                               
  %348 = (bf16[1,8], u32[1,8]) CustomCall(AwsNeuronTopK, backend_config="8")(%341)                                                                             
  %350 = u32[1,8] get-tuple-element(%348)   # indices (the top_k values are discarded)                                                                         
  %351 = s64[1,8] convert(%350)             # cast to int64 for downstream gathers                                                                             
  - The top‑k values (bf16) are not used — only the indices.                                                                                                   
  - Top‑k is taken on bf16 router logits (pre‑softmax), matching torch.topk(router_logits, top_k) in RouterTopK.forward.                                       
                                                                                                                                                               
  5c. Softmax over all 128 logits — FP32                                                                                                                       
                                                                                                                                                               
  apply_activation_fn requests dtype=torch.float64; Trainium downgrades to FP32, so the actual silicon computation is fp32:                                    
                                                                                                                                                               
  %380 = f32[1,128] convert(%341)                          # bf16 logits → fp32                                                                                
  %386 = f32[1]     reduce(%380, -inf, dims=[1], MAX)      # max for stability                                                                                 
  %388 = f32[1,128] subtract(%380, bcast(%386))                                                                                                                
  %389 = f32[1,128] exponential(%388)                                                                                                                          
  %395 = f32[1]     reduce(%389, 0.0, dims=[1], ADD)                                                                                                           
  %397 = f32[1,128] divide(%389, bcast(%395))              # softmax probs                                                                                     
  %398 = bf16[1,128] convert(%397)                         # cast to bf16 before gather                                                                        

  NOTE (confirmed via NTFF): in the compiled graph these five HLO ops are fused into a single custom-call
  on %convert.380 with op_type="aten__softmax". The hardware realisation uses a dedicated Scalar+Vector
  engine sequence (ACT_TABLE_LOAD → TENSOR_REDUCE(MAX) → ACTIVATE COPY(scale/bias) → ACTIVATE EXP →
  TENSOR_REDUCE(SUM) → RECIPROCAL → ACTIVATE COPY(scale/bias)). All arithmetic is fp32 in the
  activation/vector pipelines. See Section 10.2 for the exact trace.
                                                                                                                                                               
  5d. Gather top‑k softmax affinities — bf16                                                                                                                   
                                                                                                                                                               
  Builds 2‑D gather indices (batch_idx, topk_pos) and gathers the bf16 softmax output:                                                                         
                                                                  
  %379 = s64[1,8,2] concatenate(%377, %378)                # [batch_idx, expert_idx] per top-k                                                                 
  %399 = bf16[1,8] gather(%398, %379)                      # chosen_expert_affinities                                                                          
                                                                                                                                                               
  5e. L1 normalization of chosen affinities — bf16 (this matters for bit accuracy)                                                                             
                                                                                                                                                               
  This is F.normalize(x, p=1.0, dim=1, eps=1e-12), executed entirely in bf16:                                                                                  
                                                                  
  %400 = bf16[1,8] abs(%399)                                                                                                                                   
  %407 = bf16[1]   reduce(%400, 0, dims=[1], ADD)          # bf16 accumulator!                                                                                 
  %411 = bf16[1,1] clamp(min=%318, val=%408, max=%317)                                                                                                         
         where %318 = 0x2B8D ≈ 1.001e-12  (bf16 of default eps=1e-12)                                                                                          
               %317 = 0x7F80 = +Inf        (no upper bound)                                                                                                    
  %415 = bf16[1,8] divide(%399, broadcast(%411))           # normalized chosen_expert_affinities                                                               
                                                                                                                                                               
  Take note: the reduction is AddComputation.403 which is bf16 add, not FP32. Summing 8 bf16 values in bf16 is a non‑associative source of drift.              
                                                                                                                                                               
  After normalization, the 8 scalars are cast to FP32 for later use:                                                                                           
  %418 = f32[8,1] convert(%417=bf16[8,1] reshape(%415))           
                                                                                                                                                               
  5f. Gather per‑expert weight slabs                                                                                                                           
                                                                                                                                                               
  forward_selective_loading reads only the 8 chosen experts' weights from HBM (per rank):                                                                      
                                                                                                                                                               
  %431 = s64[8,1] concatenate(top_k_indices)               # after negative-index canonicalization                                                             
  %433 = bf16[8, 192, 2048] gather(%432=down_proj, %431)   # down_proj for 8 experts                                                                           
  %449 = bf16[8, 2048, 384] gather(%448=gate_up,  %431)    # gate_up_proj for 8 experts                                                                        
  - slice_sizes=[1, 192, 2048] (down), [1, 2048, 384] (gate_up). Each gather pulls exactly the 8 chosen expert slabs from the 128 stored experts.              
                                                                                                                                                               
  5g. Gate/Up projection — bf16 matmul with fp32 PSUM                                                                                                          
                                                                                                                                                               
  %464 = bf16[1,1,2048] reduce(%454 = bf16[1,1,1,2048], 0, dims=[0], ADD)   # identity (reducing dim=1)
  %465 = bf16[1,1,8,384] dot(%464, %449)                                     # contract: lhs=[2], rhs=[1]                                                      
  %466 = bf16[8,1,1,384] transpose(%465)                                     # move expert axis to front                                                       
  - Shape math: [1,1,2048] · [8,2048,384] → [1,1,8,384]. LHS is broadcast (no batch dim); each expert receives the same hidden vector.                         
  - bf16 × bf16 → bf16, with FP32 accumulator (mixed‑precision flag).                                                                                          
                                                                                                                                                               
  5h. Activation (GLU + SiLU) — sliced, cast‑heavy                                                                                                             
                                                                                                                                                               
  Mapping to Experts._activation, with scaling_factor=1.0, bias=0.0:                                                                                           
                                                                                                                                                               
  # gate = x[..., 0:192], up = x[..., 192:384]   (torch.chunk order)                                                                                           
                                                                                                                                                               
  # UP path (up + bias, bias=0):                                                                                                                               
  %467 = bf16[8,1,1,192] slice(%466, last_dim=[192:384])        # UP                                                                                           
  %468 = bf16[] multiply(%435=0.0_bf16, %434=1.0_bf16)          # = 0.0 (compile-time folded bias expression)                                                  
  %470 = bf16[8,1,1,192] add(%467, broadcast(%468))             # up + 0  (still a real bf16 add in HLO)                                                       
  %471 = f32[8,1,1,192] convert(%470)                           # → fp32                                                                                       
                                                                                                                                                               
  # GATE path (scaling_factor * gate, scaling=1.0):                                                                                                            
  %473 = bf16[8,1,1,192] slice(%466, last_dim=[0:192])          # GATE                                                                                         
  %474 = f32[8,1,1,192] convert(%473)                           # → fp32                                                                                       
  %476 = f32[8,1,1,192] multiply(%474, broadcast(%472=1.0_f32)) # gate * 1.0                                                                                   
  %477 = bf16[8,1,1,192] convert(%476)                          # → bf16 for the SiLU custom call                                                              
                                                                                                                                                               
  # SiLU:                                                                                                                                                      
  %484 = bf16[8,1,1,192] CustomCall(AwsNeuronSilu)(%477)        # silu in bf16                                                                                 
                                                                                                                                                               
  # Combine (silu(gate) * up) in fp32:                                                                                                                         
  %485 = f32[8,1,1,192] convert(%484)                           # silu → fp32                                                                                  
  %486 = f32[8,1,1,192] multiply(%485, %471)                    # silu(gate) * up  (fp32)                                                                      
  %487 = bf16[8,1,1,192] convert(%486)                          # → bf16 for down_proj input                                                                   
  Key precisions:                                                                                                                                              
  - The bias add and the slicing happen in bf16.                                                                                                               
  - The scale×gate multiply happens in fp32 (because hidden_act_scaling_factor lowers as an fp32 constant), but the result is cast to bf16 before the SiLU.    
  - SiLU is the AwsNeuronSilu custom op on bf16 inputs; its internal math is bf16 (NeuronCore activation engine on bf16; for bit accuracy you should match the 
  ACT‑engine SiLU on bf16, not an fp32 reference).                                                                                                             
  - The gate×up combine is fp32; the product is rounded to bf16 before down_proj.                                                                              
                                                                                                                                                               
  5i. Down projection — bf16 batched matmul                                                                                                                    
                                                                                                                                                               
  %488 = bf16[8,1,1,2048] dot(%487, %433)                                                                                                                      
           lhs_contract=[3], rhs_contract=[1], lhs_batch=[0], rhs_batch=[0]                                                                                    
  - [8,1,1,192] · [8,192,2048] → [8,1,1,2048]: batched over the expert dim, contract on I_TP=192.                                                              
  - bf16 × bf16 → bf16, FP32 accumulator.                                                                                                                      
                                                                                                                                                               
  5j. Per‑expert weighting and sum‑reduce over top‑k — bf16 reduction!                                                                                         
                                                                                                                                                               
  %491 = bf16[8,2048] reshape(%490)
  %492 = f32[8,2048]  convert(%491)                      # → fp32                                                                                              
  %493 = f32[8]       reshape(%418)                      # normalized chosen affinities (fp32)                                                                 
  %495 = f32[8,2048]  multiply(%492, broadcast(%493))    # per-expert output * affinity, fp32                                                                  
  %496 = bf16[8,2048] convert(%495)                      # round to bf16                                                                                       
  %497 = bf16[] constant(0.0)                                                                                                                                  
  %503 = bf16[2048]   reduce(%496, %497, dims=[0], AddComputation.499)                                                                                         
  Two very important bit‑accuracy facts here:                                                                                                                  
  1. The affinity modulation out_e * a_e is done in FP32 (convert up, multiply, then convert back to bf16).                                                    
  2. The top‑k reduction sum_{e in top_k} (AddComputation.499 = bf16 add at the HLO level) is physically
     implemented as reduce‑as‑MATMUL on the Tensor engine — i.e. a matmul against an all-ones vector with
     an FP32 PSUM accumulator, with the final value rounded to bf16 once on store. Effective semantics:
       out[h] = bf16( Σ_{e=0..7} fp32( wz_bf16[e, h] ) )
     not a stepwise bf16 accumulator. Match this in your kernel with a single fp32-accumulated reduction
     rounded once at the end — do NOT fold left/right in bf16 and do NOT keep the running sum in fp32
     across iterations without matching the rounding semantics of a one-shot reduction. See Section 10.3
     for the hardware evidence (`%reduce.503` lowers to TENSOR_REDUCE + MATMUL with acc_flags=3).

  This is a correction to the earlier HLO-only reading, which suggested a literal bf16 sequential sum.
                                                                                                                                                               
  5k. TP all‑reduce — bf16                                                                                                                                     
                                                                                                                                                               
  %515 = (bf16[1,1,2048], bf16[]) all-reduce(%513=moe_output_bf16, token_bf16)
         replica_groups=[[0,1,2,3]], constrain_layout=true, channel_id=0                                                                                       
         computation = AddComputation.509 (bf16 add)                                                                                                           
  - Reduce op is bf16 sum across the 4 TP ranks.                                                                                                               
  - Because disable_numeric_cc_token=True, the fused numeric compensator token that openxla sometimes attaches is disabled. The (data, token) tuple output is  
  still present, but the %514 token is effectively unused for numerics. Don't try to use it to "fix" reduction error.                                          
  - After all‑reduce, %516 = bf16[1,1,2048] is reshaped and added to the pre‑MoE residual in the decoder wrapper.                                              
                                                                                                                 
  6. End‑to‑end summary (per layer, TKG, per TP rank)                                                                                                          
                                                                                                                                                               
  Let x ∈ bf16^H, H=2048, I_TP=192, E=128, k=8. Pseudocode matched to the HLO:                                                                                 
                                                                                                                                                               
  # --- post_attention_layernorm (outside mlp) ---                                                                                                             
  xf = fp32(x)                                                                                                                                                 
  n  = RMSNorm_custom(xf, weight_bf16, eps=1e-6_f32)    # internal math fp32
  h  = bf16(n)                                          # MoE input                                                                                            
                                                                  
  # --- Router ---                                                                                                                                             
  logits_bf16 = h @ Wr          ; Wr: bf16[2048,128]    # fp32 PSUM, bf16 result
  topk_vals_bf16_UNUSED, idx_u32 = AwsNeuronTopK(logits_bf16, k=8)                                                                                             
  idx_s64 = s64(idx_u32)                                                                                                                                       
                                                                                                                                                               
  # --- Softmax (full, fp32) ---                                                                                                                               
  logits_f32 = fp32(logits_bf16)                                                                                                                               
  m          = max(logits_f32, dim=-1, keepdim=True)              
  p_f32      = exp(logits_f32 - m) / sum(exp(logits_f32 - m), dim=-1, keepdim=True)                                                                            
  p_bf16     = bf16(p_f32)                                                                                                                                     
                                                                                                                                                               
  # --- Gather top-k affinities and L1-normalize (all bf16) ---                                                                                                
  a_bf16     = gather(p_bf16, idx_s64)                    # bf16[k]                                                                                            
  s          = clamp(sum_bf16(|a_bf16|), min=bf16(1e-12), max=+inf)   # bf16 accumulator                                                                       
  a_norm_bf16= a_bf16 / s                                                                                                                                      
  a_norm_f32 = fp32(a_norm_bf16)                          # kept as fp32 for the final weighting                                                               
                                                                                                                                                               
  # --- Selective weight load (bf16) ---                                                                                                                       
  Wgu = gather(gate_up_proj, idx_s64)    ; Wgu: bf16[k, H, 2*I_TP=384]  # [gate|up]                                                                            
  Wdn = gather(down_proj,    idx_s64)    ; Wdn: bf16[k, I_TP, H]                                                                                               
                                                                                                                                                               
  # --- Fused gate/up projection ---                                                                                                                           
  y = h @ Wgu                             ; y:   bf16[1,1,k,384]   # fp32 PSUM                                                                                 
  y = transpose(y)                        ; y:   bf16[k,1,1,384]                                                                                               
  gate = y[..., 0:192]   ; up = y[..., 192:384]                      # bf16 each                                                                               
                                                                                                                                                               
  # --- GLU activation (precision schedule exactly as emitted) ---                                                                                             
  up_plus_b   = bf16( up + bf16(0.0) )                               # bias add happens in bf16                                                                
  up_f32      = fp32(up_plus_b)                                                                                                                                
  gate_scaled = bf16( fp32(gate) * fp32(1.0) )                       # scaling in fp32, round to bf16
  silu_bf16   = AwsNeuronSilu(gate_scaled)                           # bf16 custom call                                                                        
  act_bf16    = bf16( fp32(silu_bf16) * up_f32 )                     # combine in fp32, round to bf16                                                          
                                                                                                                                                               
  # --- Down projection ---                                                                                                                                    
  z_bf16 = batched_matmul(act_bf16, Wdn)      ; z: bf16[k,1,1,H]     # fp32 PSUM                                                                               
  z_bf16 = reshape(z_bf16)                    ; z: bf16[k, H]                                                                                                  
                                                                                                                                                               
  # --- Expert-weighted sum over top-k (bit-sensitive) ---                                                                                                     
  z_f32   = fp32(z_bf16)                                                                                                                                       
  wz_f32  = z_f32 * a_norm_f32[:, None]                              # fp32 multiply                                                                           
  wz_bf16 = bf16(wz_f32)                                             # round each per-expert row to bf16                                                       
  out_bf16 = bf16( sum_f32( fp32(wz_bf16), dim=0 ) )                 # one-shot fp32 reduce, round once
                                                                     # (HW: TENSOR_REDUCE + MATMUL fp32 PSUM)
                                                                                                                                                               
  # --- TP all-reduce (bf16 add) ---                                                                                                                           
  out_bf16 = all_reduce_bf16_sum(out_bf16, group=[0,1,2,3])          # bf16 sum across 4 ranks                                                                 
  # Residual add + next layer...                                                                                                                               
                                                                                                                                                               
  7. Bit‑accuracy checklist for your kernel                                                                                                                    
                                                                                                                                                               
  To match this compiled graph bit‑exactly when using the same rank’s shard of weights:                                                                        
  
  1. Router must be computed as bf16 @ bf16 with fp32 accumulator and bf16 store. Do not promote the router weight to fp32 even though router_config.dtype says
   so — the HLO parameter is bf16.                                                                                                                           
  2. Top‑k is taken on bf16 pre‑softmax logits. Do not apply softmax before topk.                                                                              
  3. Softmax is fp32 (not fp64, and not bf16). Use the max‑subtract stabilization exactly as emitted. Cast result to bf16 before gather.                       
  4. L1 normalization of the top‑k affinities is entirely bf16 with eps ≈ bf16(1e-12) = 0x2B8D. Sum of 8 bf16 values uses bf16 accumulator. Do not reorder or  
  promote.                                                                                                                                                     
  5. After normalization, the affinities are held as fp32 for the final per‑expert multiply. Emit the bf16 → fp32 cast explicitly.                             
  6. GLU activation must follow the exact cast schedule:                                                                                                       
    - UP path: bf16 add bias(=0) → fp32.                                                                                                                       
    - GATE path: fp32 multiply scale(=1) → bf16 → AwsNeuronSilu → fp32.                                                                                        
    - Combine: fp32 × fp32 → bf16.                                                                                                                             
  The bias and scaling constants fold to 0.0 and 1.0 respectively for Qwen3‑MoE, but the casts remain and change rounding.                                     
  7. SiLU must be the bf16 AwsNeuronSilu custom op. An fp32 silu will produce drift. On hardware this is the activation engine's bf16 SiLU.                    
  8. Down proj: bf16 batched dot (lhs_batch=[0], rhs_batch=[0]) with fp32 PSUM.                                                                                
  9. Per‑expert weighting is fp32 → round each per-expert row to bf16. The top‑k sum is then done on the
  Tensor engine as a reduce‑as‑matmul with FP32 PSUM and a single bf16 rounding on store. i.e.:
       wz_bf16[e] = bf16( fp32(z_bf16[e]) * a_f32[e] )
       out_bf16   = bf16( Σ_e fp32(wz_bf16[e]) )
  Do NOT replace the rounding‑then‑reduce with a fused fp32 mul-accumulate; the per-expert bf16 rounding
  happens before the reduction and is load-bearing. Do NOT implement the top-k sum as a stepwise bf16
  accumulator either — it is a one-shot fp32-accumulated reduce.
  10. All‑reduce across the 4 TP ranks is a bf16 sum with replica_groups [[0,1,2,3]], constrain_layout=True, channel_id=0, and disable_numeric_cc_token=True   
  (so the numeric‑compensator token is off; don't add one).                                                                                                    
  11. Weight slab gather is exactly gather(gate_up, idx) and gather(down, idx) of slice_sizes [1, 2048, 384] and [1, 192, 2048] respectively — same gather   
  pattern on every rank; differences come only from the rank's weight shard.                                                                                   
  12. post_attention_layernorm is not inside the MoE module for this config — it is applied by the decoder wrapper: bf16 → fp32 → AwsNeuronRmsNorm(fp32,     
  weight=bf16, eps=1e-6_f32) → bf16. Eps is an fp32 constant; weight is bf16 and the multiply happens inside the custom op at fp32 precision (then cast to fp32
   output, then bf16 by the outer convert).                                                                                                                  
                                                                                                                                                               
  8. Files / ops to mirror                                                                                                                                     
  
  - MoE._forward_compute_bound — driver (no rmsnorm, no shared experts, no SP, no EP, no token shuffle).                                                       
  - ExpertMLPsV2.forward_selective_loading — top-k-indexed Experts(...) invocation (T=1).                                                                    
  - Experts.forward / Experts._activation (glu_type=GLU, silu, scaling=1, bias=0).                                                                             
  - RouterTopK.forward → RouterBase.get_router_logits (bf16) + apply_activation_fn (fp64 requested → fp32 on device) + torch.topk on bf16 logits.              
  - nn.Linear of router: weight [E,H], stored bf16, replicated across TP.                                                                                      
  - RMSNorm: torch_neuronx.xla_impl.ops.RmsNorm → AwsNeuronRmsNorm(input_f32, weight_bf16, eps_f32, dim=-1), wrapped by CustomRMSNorm for the bf16↔fp32 dance. 
                                                                                                                                                               
  If you reproduce the above op sequence with exactly these dtypes, casts, reduction orders, and custom‑op choices on the same weight shards, your kernel      
  output will match the compiled graph bit‑for‑bit at the MoE boundary on every TP rank.            

---

# NEFF/NTFF (hardware execution)

Ground-truth mapping of the MoE HLO ops to actual hardware instructions, derived from the compiled NEFF
and runtime NTFF (`neuron-profile view --output-format json`). Evidence pulled from:

- NEFF: `output/.../neff_138311351493953_vnc_2.neff`
  (corresponds to `baseline_compiler_dir/token_generation_model/_tp0_bk2/`).
- NTFF: `output/.../138311351493953_vnc_0.ntff` (NC 4 of an LNC=2 pair).

All counts/times below are for **one decoder layer's MoE section on one NC** (`MoE[.49][97]`). Layer 0
appears in the compiled module under identifier `[.49]` (subsequent layers are [.50], [.51], ...).
1,859 hardware instructions live inside this MoE block on a single NC.

Aggregate profile context:
- instance_type = trn3.3xlarge, LNC=2 (sg00=NC4, sg01=NC5).
- total_time = 13.24 ms; MFU=0.08%; MBU=10.4% (bandwidth-bound, as expected for TKG).
- HBM read 1.61 GB, write 7.17 MB across the whole model step.
- 99 CC ops total (1 BARRIER + 98 bf16 AllReduces of 2048 elements each).

## 10.1 HLO → hardware-op map per MoE layer (one NC)

| HLO op (attribution)                      | Engine(s)        | Hardware opcodes                                                          |  Count |   Active |     HBM read |
|-------------------------------------------|------------------|---------------------------------------------------------------------------|-------:|---------:|-------------:|
| `%dot.9`  = router matmul `h @ Wrᵀ`       | Tensor           | LDWEIGHTS ×32, MATMUL ×32                                                 |     74 |  8.77 μs |    1024 KB   |
| `%custom-call.444` = AwsNeuronTopK        | Vector           | MAX8 ×2, MATCH_VALUE_LOAD ×2, FIND_INDEX8 ×2 (+ 4 Sync/DMA)               |     10 |  2.84 μs |        —     |
| `%custom-call = ...(%convert.380)` = softmax (compiler-fused) | Scalar + Vector | ACT_TABLE_LOAD, TENSOR_REDUCE ×2, ACTIVATE COPY(scale/bias) ×2, ACTIVATE EXP, RECIPROCAL (+ sync/DMA) | 14 | 4.41 μs | 0.5 KB |
| `%gather.399` = top-k gather of softmax probs | GpSimd       | LOAD_MASK_SELECT, POOL_BUFFER_LOAD, STREAM_SHUFFLE, GATHER, DMA_DIRECT2D  |      6 |  1.26 μs |     0.2 KB   |
| `%add.441/357`, `%select.445/361` = neg-index canonicalization | GpSimd | TENSOR_TENSOR, COPY, COPY_PREDICATED, COPY_PREDICATED_SCALAR          |    ~22 |  ~3.8 μs |        —     |
| `%clamp.0` = F.normalize clamp (ε=1e-12)  | Scalar           | TENSOR_SCALAR                                                             |      1 |  0.17 μs |        —     |
| `%divide.415` = L1 normalize (a / Σ|a|)   | Vector + Scalar  | RECIPROCAL, ACTIVATE                                                      |      2 |  0.48 μs |        —     |
| `%gather.449` = gather(gate_up_proj, idx) | GpSimd + DMA     | ALU_OP ×40, TENSOR_LOAD ×16, MOVE ×16, **DMA_DIRECT2D ×16**               |     89 |  9.47 μs | **12,288 KB (12 MB)** |
| `%gather.433` = gather(down_proj, idx)    | GpSimd + DMA     | ALU_OP ×40, TENSOR_LOAD ×16, MOVE ×16, **DMA_DIRECT2D ×16**               |     88 |  8.19 μs | **6,144 KB (6 MB)**   |
| `%dot.10` = gate_up matmul                | Tensor           | LDWEIGHTS ×512, MATMUL ×512 (tile 128×96; acc_flags {0:472, 1:20, 2:20})  |  1,024 | 156.8 μs |        —     |
| `%custom-call.445` = AwsNeuronSilu        | Scalar + Vector  | ACT_TABLE_LOAD ×2, ACTIVATE act_fn=SILU ×2, TENSOR_TENSOR ×2              |      6 |  3.36 μs |        —     |
| `%dot.11` = down_proj matmul              | Tensor           | LDWEIGHTS ×256, MATMUL ×256 (tile 96×128; acc_flags=3 everywhere)         |    517 | 65.9 μs  |        —     |
| `%reduce.503` = top-k bf16 sum (dim=0)    | Tensor + Vector  | TENSOR_REDUCE + LDWEIGHTS + **MATMUL (transpose_mode=ENABLED)** + COPY + DMA_DIRECT2D | 5 | 1.28 μs | — |
| `%all-reduce.515` = TP all-reduce (bf16)  | GpSimd trigger → CC-cores | WRITE (GpSimd) + EVENT_SEMAPHOREs; data movement on CC-cores     |      1 | 0.23 μs trigger / **9.15 μs CC** | 4 KB |

Per-engine active time within one MoE layer on one NC (span ≈ 204 μs; engines run concurrently):

| Engine   | Instructions | Active time |
|----------|-------------:|------------:|
| Tensor   |        1,606 |    230.7 μs |
| GpSimd   |          192 |     19.2 μs |
| Scalar   |           25 |      9.1 μs |
| Vector   |           26 |      4.8 μs |
| Sync     |           10 |      3.1 μs |

## 10.2 Softmax is fused to a single custom-call on hardware

HLO has five element-wise ops (max / sub / exp / sum / div), but the compiled graph collapses them into
one custom-call tagged `op_type="aten__softmax"`. The hardware sequence, in timestamp order for
`%custom-call = custom-call(%convert.380)`:

```
[Scalar] ACT_TABLE_LOAD                           # load EXP LUT
[Sync ] DMA_DIRECT2D  × 2                         # stage
[Vector] TENSOR_REDUCE                            # MAX over 128 lanes
[Scalar] ACTIVATE act_fn="COPY(scale/bias enabled)"  # subtract max  (bias = −max)
[Scalar] ACTIVATE act_fn="EXP"                    # exp via LUT
[Vector] TENSOR_REDUCE                            # SUM
[Vector] RECIPROCAL                               # 1 / sum
[Scalar] ACTIVATE act_fn="COPY(scale/bias enabled)"  # multiply by 1/sum
```

Everything is fp32 internally (input is `%convert.380 = fp32(bf16 logits)`). There are no intermediate
bf16 tensors you can hook into — reproduce the end-to-end fp32 fused softmax, then cast to bf16.

## 10.3 Top-k bf16 reduce is actually reduce-as-matmul with fp32 PSUM

`%reduce.503 = bf16[2048] reduce(bf16[8,2048], init=0, dims=[0], AddComputation.499)` lowers to a
Tensor-engine MATMUL:

```
operands: S[5] (Tensor)++@complete transpose_mode=ENABLED acc_flags=3 psum_zero=2048
          src=bfloat16@...  dst=...  128*16
```

`transpose_mode=ENABLED` + an implicit all-ones right-hand operand makes the 8-wide reduce into a
matmul that **accumulates in FP32 PSUM** and stores the final bf16 value in a single tile.
This matters for bit-accuracy: treat the top-k sum as one-shot `bf16( Σ_{e} fp32(wz[e]) )`, not as a
chain of bf16 adds, and not as an fp32 sum that skips the per-expert bf16 rounding (that rounding is
done by the preceding `%convert.496`, which is the `fp32 * fp32 → bf16` store of the affinity-modulated
expert outputs before this reduce).

## 10.4 TopK = hardware primitive, not softmax-then-sort

`AwsNeuronTopK` has no fp32 path. On the Vector engine:

```
[Vector] MAX8          — dedicated 8-way max primitive
[Vector] MATCH_VALUE_LOAD
[Vector] FIND_INDEX8   — extracts the u32 indices of the 8 max bf16 values
(×2 rounds to cover the 128-lane dim)
```

Input is bf16 router logits, output is (bf16 values, u32 indices). The bf16 values are discarded. This
is deterministic for a given bf16 input; replace with the same primitive in your kernel (do not sort
fp32 softmax probs).

## 10.5 SiLU is the activation engine's LUT-based bf16 SILU

`AwsNeuronSilu` under `%custom-call.445 = custom-call(%slice.473)` with
`op_type="xla___op_SiluForwardImpl"`:

```
[Scalar] ACT_TABLE_LOAD                 # load SILU LUT (sigmoid table + x·sigmoid(x) fused)
[Scalar] ACTIVATE act_fn="SILU"         # hardware SILU via LUT
[Vector] TENSOR_TENSOR                  # outer fp32 combine, rounded to bf16 per HLO %487
```

Input is bf16, output is bf16. ACT-engine internals use an fp32 LUT but the rounded-to-bf16 store is
deterministic — your reference SILU must match the LUT, not a Python `silu(x.float()).to(bfloat16)`.

## 10.6 Expert weight gather = vector-dynamic-offset DGE DMA (GpSimd → HBM → SBUF)

`%gather.449` (gate_up slab) lowers to 16 PSEUDO_DMA_DIRECT2D instructions issued on GpSimd, with the
HBM source address computed on-the-fly from a register table written by `ALU_OP MULTIPLY`:

```
ALU_OP:   op=MULTIPLY dtype=int32  src0=$R[61] src1=786432  dst=$R[60]      # idx × per-expert-bytes
DMA_DIRECT2D:
  compiler_opcode = PSEUDO_DMA_DIRECT2D     dge_op = DIRECT2D
  src_elem_size   = 6144                    dst_elem_size = 6144
  src_pattern     = [6144,1][128,1]         → 128 rows × 6144 bytes = 768 KB per DMA
  dst_pattern     = [262144,1][128,1]
  src_table_offset_reg = $R[61]             ← vector_dynamic_offsets DGE
  src_table_index      = 24
  dst_addr_imm         = 0x802000017220     (SBUF)
  hbm_read_bytes  = 786432                  sbuf_write_bytes = 786432
```

- 786,432 bytes = 2048 × 192 × 2 = one expert's gate_up slab-half.
- 16 DMAs × 768 KB = 12 MB = 8 experts × 2048 × 384 × 2 bytes ✓ (matches HLO `slice_sizes`).
- Source: HBM; Destination: SBUF. Issued by GpSimd engine; data moves on DMA queues.

`%gather.433` (down_proj) is analogous: 16 DMAs × 384 KB = 6 MB.

Tiny gathers (`%gather.399`, picking 8 bf16 softmax probs by index) use a different GpSimd path
(`LOAD_MASK_SELECT + POOL_BUFFER_LOAD + STREAM_SHUFFLE + GATHER`), not DGE, because the payload is
small.

## 10.7 TP AllReduce

From `cc_ops`:

```
operation     = AllReduce
algorithm     = Mesh                          (not Ring / not Tree)
replica_group = [[0, 1, 2, 3]]
dtype         = bf16
num_elements  = 2048
input_size    = 4096 bytes (bf16 × 2048)
output_size   = 4096 bytes
trigger_engine= GpSimd                        (a single WRITE on GpSimd fires it)
duration      = ~9.15 μs  (one MoE AR)
```

On the Tensor/Vector/Scalar engines the AR appears only as a single `WRITE` (GpSimd) plus semaphore
events; the actual network traffic runs on the CC-cores. With
`--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2`, the MoE AllReduce on layer N overlaps
the next layer's attention/matmul work: verified by tracing the first MoE AR (trigger at t=231,604 ns)
— within the next 20 μs, **334 of 363** Tensor-engine instructions belong to layer 50 (and only 14 to
layer 49 itself). `cc_op_active_time_percent = 5.86%` across the whole model, i.e. most CC time is
hidden behind compute.

## 10.8 RMSNorm (context — applied outside the MoE module)

Both `input_layernorm` and `post_attention_layernorm` (CustomRMSNorm[.194] and [.197]) lower to the
same Scalar+Vector+Tensor sequence on hardware:

```
[Scalar] ACTIVATE act_fn="SQUARE"                 # x²
[Vector] TENSOR_REDUCE                            # Σ x²
[Tensor] LDWEIGHTS + MATMUL   × 4 (reduce-as-matmul for the 2048-dim sum)
[Vector] MEMSET   × 4                             # PSUM clears
[Scalar] ACTIVATE act_fn="RECIPROCAL_SQRT"        # 1 / √(mean_sq + ε)
[Scalar] ACTIVATE act_fn="COPY(scale/bias enabled)"  # × (1/rms)
[Vector] TENSOR_TENSOR                            # × weight (bf16)
```

Internal precision is fp32 across the ACTIVATE path; the ε is an fp32 constant (1e-6); the weight is
bf16 and the multiply happens at fp32 before the final bf16 store. Front the custom call by `bf16 →
fp32` and follow it by `fp32 → bf16` exactly as in the HLO.

## 10.9 Wall-time budget on one NC per MoE layer (engines concurrent)

```
Tensor engine active   : ~230 μs   (gate_up 157 + down 66 + router 9 + reduce 1 + small)
DMA active (expert W)  :  ~18 μs   (12 MB gate_up + 6 MB down, HBM→SBUF via DGE)
GpSimd                 :  ~19 μs   (index canonicalization + DMA descriptor writes)
Scalar  (ACTIVATE LUTs):   ~9 μs   (softmax EXP + silu + rms COPY/RECIP_SQRT)
Vector                 :   ~5 μs   (reductions, TENSOR_TENSOR, MAX8/FIND_INDEX8, RECIPROCAL)
Sync                   :   ~3 μs
CC   (AllReduce)       :   ~9 μs   (fully overlapped with the next layer's compute)
Wall span (one NC)     :  ~204 μs
```

The Tensor engine's matmuls (gate_up + down_proj ≈ 223 μs) are the wall. The expert-weight DMA
(~18 μs) finishes long before the matmuls do, so pure DMA optimisations will not shorten the critical
path unless you also shorten the matmul time.

## 10.10 Corrections to the earlier HLO-only reading

1. **Softmax.** HLO shows 5 separate ops; the hardware executes a single fused softmax custom-call
   (Section 10.2). Reference implementations must be end-to-end fp32 fused, with bf16 rounding only on
   the final store.

2. **Top-k bf16 sum (`%reduce.503`).** HLO's AddComputation.499 (bf16 add) lowers to a
   Tensor-engine reduce-as-matmul with FP32 PSUM and a single bf16 rounding (Section 10.3). The
   earlier checklist's "bf16 accumulator" is not literal — reproduce `bf16(Σ fp32(wz_bf16[e]))`, not a
   stepwise bf16 fold.

3. **AwsNeuronTopK.** Dedicated Vector-engine hardware primitives (MAX8/FIND_INDEX8), not a
   sort-based path (Section 10.4).

4. **Expert weight gather.** Lowers to PSEUDO_DMA_DIRECT2D with `src_table_offset_reg` =
   vector-dynamic-offset DGE (Section 10.6), issued by GpSimd. Not a generic on-chip gather and not
   pre-materialised.

5. **TP AllReduce.** Algorithm is Mesh across [[0,1,2,3]], bf16, 2048-element payload (Section 10.7).
   Already overlaps with the next layer's compute thanks to `--enable-ccop-compute-overlap`.

Everything else in the HLO analysis (router bf16 matmul, affinity softmax→gather→clamp→divide chain,
GLU cast schedule, down_proj batched-over-experts matmul, fp32 affinity modulation with bf16 rounding,
decoder-wrapper residual add) is confirmed by the profile.