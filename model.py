import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
from transformers import GenerationMixin
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from vq import RQVAE
from layers import *

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionResidualBlock(nn.Module):
    """Residual block with time conditioning for diffusion models."""
    def __init__(self, in_dim, out_dim, time_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim)
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        if in_dim != out_dim:
            self.residual = nn.Linear(in_dim, out_dim)
        else:
            self.residual = nn.Identity()

    def forward(self, x, t):
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.silu(h)
        t_emb = self.time_mlp(t)
        h = h * (1 + t_emb)
        h = self.fc2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        return F.silu(h + self.residual(x))


class DiffusionAdapter(nn.Module):
    """
    Diffusion-Enhanced Adapter with Learnable Gate.
    Supports both 2D (batch, dim) and 3D (batch, seq_len, dim) input.
    NOTE (v3): Internal reg_loss is still computed but callers should NOT
    add it to the training loss. It is returned for logging only.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, num_steps=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.num_steps = num_steps

        # === Original MLP path (always active for stability) ===
        self.original_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # === Diffusion path ===
        time_dim = self.hidden_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([
            DiffusionResidualBlock(self.hidden_dim, self.hidden_dim, time_dim, dropout)
            for _ in range(num_steps)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, output_dim)
        )

        # === Learnable gate ===
        self.diffusion_gate = nn.Parameter(torch.tensor(0.3))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        is_3d = (x.dim() == 3)

        if is_3d:
            x_2d = x.reshape(-1, self.input_dim)
        else:
            x_2d = x

        num_elements = x_2d.shape[0]
        device = x_2d.device

        original_out = self.original_mlp(x_2d)

        h = self.input_proj(x_2d)
        for i, block in enumerate(self.blocks):
            t = self.time_embedding(torch.full((num_elements,), i, device=device, dtype=torch.float))
            h = block(h, t)
        diffusion_out = self.output_proj(h)

        gate = torch.sigmoid(self.diffusion_gate)
        output_2d = (1 - gate) * original_out + gate * diffusion_out

        # Internal reg loss — NOT to be used in training (v3 policy)
        reg_loss = F.mse_loss(diffusion_out, original_out.detach()) * 0.01

        if is_3d:
            output = output_2d.view(original_shape)
        else:
            output = output_2d
        return output, reg_loss


class MultiScaleDiffusionAdapter(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=None, num_steps=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dims = hidden_dims or [input_dim // 2, input_dim, input_dim * 2]
        self.hidden_dims = hidden_dims
        self.num_scales = len(hidden_dims)

        self.original_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        self.scale_adapters = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else output_dim
            self.scale_adapters.append(
                DiffusionAdapter(in_dim, out_dim, in_dim, num_steps=num_steps, dropout=dropout)
            )

        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_dims[i + 1] if i < len(hidden_dims) - 1 else output_dim, output_dim)
            for i in range(len(hidden_dims))
        ])

        self.scale_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(output_dim * self.num_scales),
            nn.Linear(output_dim * self.num_scales, output_dim),
            nn.Dropout(dropout)
        )

        self.path_gate = nn.Parameter(torch.tensor(0.3))
        self.scale_gates = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        is_3d = (x.dim() == 3)

        if is_3d:
            x_2d = x.reshape(-1, self.input_dim)
        else:
            x_2d = x

        num_elements = x_2d.shape[0]

        original_out = self.original_mlp(x_2d)

        scale_outputs = []
        total_reg_loss = torch.tensor(0.0, device=x_2d.device)
        h = self.input_proj(x_2d)
        for i, adapter in enumerate(self.scale_adapters):
            h, reg_loss = adapter(h)
            total_reg_loss = total_reg_loss + reg_loss
            projected = self.scale_projections[i](h)
            scale_outputs.append(projected)

        stacked = torch.stack(scale_outputs, dim=1)
        attn_out, _ = self.scale_attention(stacked, stacked, stacked)

        weights = F.softmax(self.scale_gates, dim=0)
        weighted = (attn_out * weights.view(1, -1, 1)).sum(dim=1)

        concat_scales = torch.cat(scale_outputs, dim=-1)
        fused = self.fusion(concat_scales)

        diffusion_out = 0.5 * weighted + 0.5 * fused

        path_gate = torch.sigmoid(self.path_gate)
        output_2d = (1 - path_gate) * original_out + path_gate * diffusion_out

        # Internal reg — NOT for training (v3 policy)
        reg_loss = F.mse_loss(diffusion_out.detach(), original_out) * 0.01
        total_reg_loss = total_reg_loss + reg_loss

        if is_3d:
            output = output_2d.view(original_shape)
        else:
            output = output_2d
        return output, total_reg_loss


class DiffusionCodebookProjection(nn.Module):
    """
    Codebook-level Diffusion Projection (continuous input version).
    NOTE (v3): No reg loss returned — refinement is controlled by the gate only.
    """
    def __init__(self, hidden_size, num_steps=3, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_steps = num_steps

        time_dim = hidden_size // 2
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        self.refinement_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + time_dim, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(num_steps)
        ])

        self.register_buffer('noise_scale', torch.linspace(0.1, 0.01, num_steps))

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

        self.refine_gate = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        original = x
        original_shape = x.shape

        if x.dim() == 3:
            h = x.reshape(-1, self.hidden_size)
        else:
            h = x

        for i, refine_layer in enumerate(self.refinement_net):
            batch = h.shape[0]
            t = torch.full((batch,), i, device=h.device, dtype=torch.float)
            t_emb = self.time_embedding(t)
            h_with_t = torch.cat([h, t_emb], dim=-1)
            delta = refine_layer(h_with_t)
            noise_scale = self.noise_scale[i]
            if self.training:
                noise = torch.randn_like(h) * noise_scale * 0.1
            else:
                noise = 0
            h = h + delta + noise
        h = self.output_layer(h)

        if len(original_shape) > 2:
            h = h.view(original_shape)

        gate = torch.sigmoid(self.refine_gate)
        output = (1 - gate) * original + gate * h
        return output


# ============================================================================
# Hierarchical Semantic Refinement Module
# ============================================================================

class HierarchicalSemanticRefinementModule(nn.Module):

    def __init__(self, codebook_size, hidden_size, diffusion_steps=4,
                 codebook_steps=3, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.base_embedding = nn.Embedding(codebook_size, hidden_size)

        self.multiscale_adapter = MultiScaleDiffusionAdapter(
            input_dim=hidden_size,
            output_dim=hidden_size,
            hidden_dims=[hidden_size // 2, hidden_size, hidden_size * 2],
            num_steps=diffusion_steps,
            dropout=dropout,
        )

        self.codebook_projection = DiffusionCodebookProjection(
            hidden_size=hidden_size,
            num_steps=codebook_steps,
            dropout=dropout,
        )

        self.simple_embedding = nn.Embedding(codebook_size, hidden_size)

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, token_ids, collect_debug: bool = False):

        simple_emb = self.simple_embedding(token_ids)
        zero_loss = torch.tensor(0.0, device=simple_emb.device)

        # Get base continuous representations
        base_emb = self.base_embedding(token_ids)

        # Stage 1: Multi-scale diffusion refinement
        refined_emb, _hsrm_reg = self.multiscale_adapter(base_emb)
        # ^^^ _hsrm_reg is DISCARDED (v3 policy: no HSRM reg in training loss)

        # Stage 2: Codebook-level diffusion refinement
        final_refined = self.codebook_projection(refined_emb)

        # Dynamic gate fusion
        gate_input = torch.cat([final_refined, simple_emb], dim=-1)
        gate = self.gate_network(gate_input)
        output = gate * final_refined + (1 - gate) * simple_emb

        debug_info = None
        if collect_debug:
            debug_info = {
                "simple_emb": simple_emb.detach(),
                "fused_emb": output.detach(),
                "gate_scalar": gate.mean(dim=-1, keepdim=True).detach(),
                "path_gate": torch.sigmoid(self.multiscale_adapter.path_gate).detach(),
                "refine_gate": torch.sigmoid(self.codebook_projection.refine_gate).detach(),
                "scale_gates": torch.softmax(
                    self.multiscale_adapter.scale_gates.detach(), dim=0
                ),
            }

        return output, zero_loss, debug_info

    def weight(self):
        """Return the base embedding weight for code_logits compatibility."""
        return self.base_embedding.weight


# ============================================================================
# Latent Diffusion Bridge
# ============================================================================

class LatentDiffusionBridge(nn.Module):


    def __init__(self, encoder_dim, decoder_dim, latent_dim, num_steps=4,
                 gap_margin=0.1):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.latent_dim = latent_dim
        self.gap_margin = gap_margin

        # Encoder to latent projection
        self.enc_to_latent = nn.Sequential(
            nn.Linear(encoder_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # Decoder to latent projection
        self.dec_to_latent = nn.Sequential(
            nn.Linear(decoder_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # Diffusion bridge in shared latent space
        self.bridge = DiffusionAdapter(
            latent_dim, latent_dim, latent_dim, num_steps=num_steps
        )

        # Output projections (back to original dimensions)
        self.enc_output = nn.Linear(latent_dim, encoder_dim)
        self.dec_output = nn.Linear(latent_dim, decoder_dim)

        # === Learnable gates — RAISED to 0.5 (sigmoid ≈ 0.62) ===
        self.enc_gate = nn.Parameter(torch.tensor(0.5))
        self.dec_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, enc_features, dec_features, collect_debug: bool = False):

        # ---- Project to shared latent space ----
        enc_latent = self.enc_to_latent(enc_features)
        dec_latent = self.dec_to_latent(dec_features)

        # Pre-bridge distance (what we want to reduce)
        pre_distance = torch.norm(enc_latent - dec_latent, dim=-1)  # (batch,)

        # ---- Bridge through diffusion ----
        combined_latent = (enc_latent + dec_latent) / 2.0
        bridged_latent, _bridge_reg = self.bridge(combined_latent)
        # ^^^ _bridge_reg is internal to DiffusionAdapter; we do NOT add it
        #     to the training loss (v3 policy).

        # ---- Project back to original dimensions ----
        raw_bridged_enc = self.enc_output(bridged_latent)
        raw_bridged_dec = self.dec_output(bridged_latent)

        # ---- Learnable gate fusion ----
        enc_gate = torch.sigmoid(self.enc_gate)
        dec_gate = torch.sigmoid(self.dec_gate)
        bridged_enc = (1 - enc_gate) * enc_features + enc_gate * raw_bridged_enc
        bridged_dec = (1 - dec_gate) * dec_features + dec_gate * raw_bridged_dec

        # ---- Post-bridge distance (after full pipeline including gating) ----
        post_enc_latent = self.enc_to_latent(bridged_enc)
        post_dec_latent = self.dec_to_latent(bridged_dec)
        post_distance = torch.norm(post_enc_latent - post_dec_latent, dim=-1)  # (batch,)

        # ================================================================
        # CORE LOSS: Semantic Gap Hinge
        #   bridge_gap_loss = mean( relu(post - pre + margin) )
        #   - When post <= pre - margin: loss = 0  (good, bridge reduced gap)
        #   - When post > pre: loss > 0  (penalize, push bridge to reduce gap)
        # ================================================================
        bridge_gap_loss = F.relu(post_distance - pre_distance + self.gap_margin).mean()

        # ---- Debug info for visualization ----
        debug_info = None
        if collect_debug:
            debug_info = {
                "pre_distance": pre_distance.detach(),    # (batch,)
                "post_distance": post_distance.detach(),   # (batch,)
                "enc_gate": enc_gate.detach(),
                "dec_gate": dec_gate.detach(),
            }

        return bridged_enc, bridged_dec, bridge_gap_loss, debug_info



@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    rank_logits: Optional[torch.FloatTensor] = None
    seq_latents: Optional[torch.FloatTensor] = None
    seq_project_latents: Optional[torch.FloatTensor] = None
    dec_latents: Optional[torch.FloatTensor] = None
    bridge_gap_loss: Optional[torch.FloatTensor] = None   # v3: renamed from diffusion_loss
    debug_info: Optional[dict] = None


class Model(nn.Module, GenerationMixin):


    def __init__(self, config, model, n_items, code_length=1, code_number=256):
        super().__init__()
        self.model = model
        self._supports_cache_class = getattr(model, '_supports_cache_class', True)
        self.config = model.config
        self.base_model_prefix = "model"
        self.generation_config = model.generation_config
        self.main_input_name = getattr(model, 'main_input_name', 'input_ids')
        self.get_encoder = model.get_encoder
        self.device = model.device
        self.can_generate = lambda: True

        self.hidden_size = model.config.hidden_size
        self.semantic_hidden_size = config.get('semantic_hidden_size')
        self.n_items = n_items
        self.code_length = code_length
        self.code_number = code_number
        self.num_beams = config['num_beams']

        # Diffusion-specific configuration
        self.diffusion_steps = config.get('diffusion_steps', 4)
        self.codebook_diffusion_steps = config.get('codebook_diffusion_steps', 3)

        # Semantic embedding (frozen, pre-trained)
        self.semantic_embedding = nn.Embedding(self.n_items, self.semantic_hidden_size)
        self.semantic_embedding.requires_grad_(False)

        # Token Embeddings: HSRM (one per code position)
        self.token_embeddings = nn.ModuleList([
            HierarchicalSemanticRefinementModule(
                codebook_size=self.code_number,
                hidden_size=self.hidden_size,
                diffusion_steps=self.diffusion_steps,
                codebook_steps=self.codebook_diffusion_steps,
            )
            for _ in range(self.code_length)
        ])
        self.token_embeddings.requires_grad_(True)

        # Encoder Adapter: Simple MLP
        e_dim = config['e_dim']
        enc_adapter_layers = [self.hidden_size] + [e_dim]
        self.enc_adapter = MLPLayers(layers=enc_adapter_layers)

        # Decoder Adapter: Simple MLP
        dec_adapter_layers = [self.hidden_size] + [self.semantic_hidden_size]
        self.dec_adapter = MLPLayers(layers=dec_adapter_layers)

        # Latent Diffusion Bridge
        gap_margin = config.get('bridge_gap_margin', 0.1)
        self.latent_bridge = LatentDiffusionBridge(
            encoder_dim=e_dim,
            decoder_dim=self.semantic_hidden_size,
            latent_dim=max(e_dim, self.semantic_hidden_size),
            num_steps=self.diffusion_steps,
            gap_margin=gap_margin,
        )

        self.apply(self._init_weights)

        self.collect_visualization = False

    def set_visualization_mode(self, enabled: bool = True):
        self.collect_visualization = bool(enabled)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None,
                                       encoder_outputs=None, **kwargs):
        return {
            "decoder_input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }

    def _shift_right(self, input_ids):
        pad_token_id = self.config.pad_token_id
        shifted_input_ids = torch.full(
            input_ids.shape[:-1] + (1,), pad_token_id, device=input_ids.device
        )
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids], dim=-1)
        return shifted_input_ids

    def get_input_embeddings(self, input_ids, attention_mask,
                             collect_debug: bool = False):
        """
        Get input embeddings using HSRM.
        v3: HSRM reg_loss is discarded (not returned to caller).
        """
        attention_mask_flatten = attention_mask.to(self.device).reshape(-1).bool()
        inputs_embeds = torch.zeros(
            *input_ids.shape, self.hidden_size, device=self.device
        )

        input_ids_clean = input_ids.clone()
        input_ids_clean[input_ids_clean == -1] = 0

        debug_info = None
        if collect_debug:
            simple_embeds = torch.zeros_like(inputs_embeds)
            gate_scalars = torch.zeros(
                input_ids.shape[0], input_ids.shape[1], 1, device=self.device
            )
            path_gates = []
            refine_gates = []
            scale_gates_list = []

        for i in range(self.code_length):
            token_ids = input_ids_clean[:, i::self.code_length]
            emb, _reg_loss, token_debug = self.token_embeddings[i](
                token_ids,
                collect_debug=collect_debug,
            )
            # _reg_loss is DISCARDED (v3: no HSRM reg in training)
            inputs_embeds[:, i::self.code_length] = emb

            if collect_debug and token_debug is not None:
                simple_embeds[:, i::self.code_length] = token_debug["simple_emb"]
                gate_scalars[:, i::self.code_length] = token_debug["gate_scalar"]
                path_gates.append(token_debug["path_gate"].view(1))
                refine_gates.append(token_debug["refine_gate"].view(1))
                scale_gates_list.append(token_debug["scale_gates"].view(1, -1))

        if collect_debug:
            valid_mask = attention_mask.to(inputs_embeds.device).unsqueeze(-1).float()
            token_count = valid_mask.sum(dim=1).clamp_min(1.0)
            sample_simple_mean = (simple_embeds * valid_mask).sum(dim=1) / token_count
            sample_fused_mean = (inputs_embeds * valid_mask).sum(dim=1) / token_count
            sample_gate_mean = (gate_scalars * valid_mask).sum(dim=1).squeeze(-1) / token_count.squeeze(-1)
            debug_info = {
                "simple_mean": sample_simple_mean.detach(),
                "fused_mean": sample_fused_mean.detach(),
                "gate_mean": sample_gate_mean.detach(),
                "path_gates": (
                    torch.cat(path_gates, dim=0).detach()
                    if len(path_gates) > 0 else torch.empty(0, device=self.device)
                ),
                "refine_gates": (
                    torch.cat(refine_gates, dim=0).detach()
                    if len(refine_gates) > 0 else torch.empty(0, device=self.device)
                ),
                "scale_gates": (
                    torch.cat(scale_gates_list, dim=0).detach()
                    if len(scale_gates_list) > 0 else torch.empty(0, device=self.device)
                ),
            }

        # Mask padded positions
        inputs_embeds = inputs_embeds.view(-1, self.hidden_size)
        inputs_embeds[~attention_mask_flatten] = self.model.shared.weight[0].to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.view(input_ids.shape[0], -1, self.hidden_size)

        return inputs_embeds, debug_info

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                labels=None, decoder_input_ids=None, decoder_inputs_embeds=None,
                encoder_outputs=None, **kwargs):
        collect_debug = bool(
            kwargs.pop("collect_debug", False) or self.collect_visualization
        )


        if input_ids is not None:
            inputs_embeds, hsrm_debug_info = self.get_input_embeddings(
                input_ids, attention_mask, collect_debug=collect_debug,
            )
        else:
            hsrm_debug_info = None

        if decoder_input_ids is None and labels is None:
            decoder_input_ids = torch.zeros(
                input_ids.size(0), self.code_length
            ).long().to(input_ids.device)
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = []
            for i in range(min(decoder_input_ids.shape[1], self.code_length)):
                if i == 0:
                    code_embedding = self.model.shared
                    decoder_inputs_embeds.append(
                        code_embedding(decoder_input_ids[:, i])
                    )
                else:
                    emb, _, _ = self.token_embeddings[i - 1](
                        decoder_input_ids[:, i],
                        collect_debug=False,
                    )
                    decoder_inputs_embeds.append(emb)
            decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)


        model_outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs,
        )


        decoder_outputs = model_outputs.decoder_hidden_states[-1]
        code_logits = []
        for i in range(min(decoder_inputs_embeds.shape[1], self.code_length)):
            centroid = self.token_embeddings[i].weight().t()
            code_logits.append(torch.matmul(decoder_outputs[:, i], centroid))
        code_logits = torch.stack(code_logits, dim=1)


        seq_latents = model_outputs.encoder_last_hidden_state.clone()
        seq_latents[~attention_mask] = 0
        seq_last_latents = torch.sum(seq_latents, dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
        seq_project_latents = self.enc_adapter(seq_last_latents)


        dec_latents = model_outputs.decoder_hidden_states[-1].clone()
        dec_latents = dec_latents[:, 0, :]
        dec_latents = self.dec_adapter(dec_latents)


        bridged_enc, bridged_dec, bridge_gap_loss, ldb_debug_info = self.latent_bridge(
            seq_project_latents, dec_latents, collect_debug=collect_debug,
        )
        seq_project_latents = bridged_enc
        dec_latents = bridged_dec


        debug_info = None
        if collect_debug:
            debug_info = {
                "hsrm": hsrm_debug_info,
                "ldb": ldb_debug_info,
            }

        outputs = QuantizeOutput(
            logits=code_logits,
            seq_latents=seq_last_latents,
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents,
            bridge_gap_loss=bridge_gap_loss,
            debug_info=debug_info,
        )
        return outputs

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 n_return_sequences: int = 1,
                 prefix_allowed_tokens_fn=None) -> torch.Tensor:
        if prefix_allowed_tokens_fn is not None:
            inputs_embeds, _ = self.get_input_embeddings(
                input_ids, attention_mask, collect_debug=False
            )
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=self.code_length + 1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        else:
            outputs = self.my_beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.code_length + 1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                return_score=False,
            )
        outputs = outputs[:, 1:].reshape(-1, n_return_sequences, self.code_length)
        return outputs

    def my_beam_search(self, input_ids, attention_mask, max_length=6,
                       num_beams=1, num_return_sequences=1, return_score=False):
        batch_size = input_ids.shape[0]

        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )

        inputs_embeds, _ = self.get_input_embeddings(
            input_ids, attention_mask, collect_debug=False
        )

        with torch.no_grad():
            encoder_outputs = self.get_encoder()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.forward(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                decoder_input_ids, beam_scores = self.beam_search_step(
                    outputs.logits, decoder_input_ids, beam_scores,
                    beam_idx_offset, batch_size, num_beams,
                )

        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True
        if return_score:
            return (
                decoder_input_ids[selection_mask.view(-1), :],
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1),
            )
        return decoder_input_ids[selection_mask.view(-1), :]

    def prepare_beam_search_inputs(self, input_ids, attention_mask, batch_size,
                                   num_beams):
        decoder_input_ids = torch.ones(
            (batch_size * num_beams, 1), device=self.device, dtype=torch.long
        )
        initial_decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))
        beam_idx_offset = (
            torch.arange(batch_size, device=self.device).repeat_interleave(num_beams) * num_beams
        )
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset

    def beam_search_step(self, logits, decoder_input_ids, beam_scores,
                         beam_idx_offset, batch_size, num_beams):
        assert batch_size * num_beams == logits.shape[0]
        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size
        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)
        decoder_input_ids = torch.cat([
            decoder_input_ids[beam_idx + beam_idx_offset, :],
            beam_next_tokens.unsqueeze(-1),
        ], dim=-1)
        return decoder_input_ids, beam_scores
