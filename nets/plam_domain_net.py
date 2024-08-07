import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip

from collections import OrderedDict
import torch.distributed as dist
import torch.autograd as autograd
import numpy as np
from .imagenet_templates import IMAGENET_TEMPLATES
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# class UniformDropout(nn.Module):
#     def __init__(self, total_dim=512, dropout_count=5):
#         super(UniformDropout, self).__init__()
#         self.total_dim = total_dim
#         self.dropout_count = dropout_count
#
#     def forward(self, x):
#         # 确保输入是期望的维度
#         if x.size(1) != self.total_dim:
#             raise ValueError(f"输入特征维度应为 {self.total_dim}, 但得到 {x.size(1)}")
#
#         # 生成掩码
#         mask = self.generate_uniform_mask(x.size(0))
#         mask = mask.to('cuda')
#         return x * mask
#
#     def generate_uniform_mask(self, batch_size):
#         mask = torch.ones(batch_size, self.total_dim)
#         for i in range(batch_size):
#             # 在0到total_dim-1中均匀随机选择dropout_count个位置
#             indices = np.linspace(0, self.total_dim - 1, self.total_dim, dtype=int)
#             dropout_indices = np.random.choice(indices, self.dropout_count, replace=False)
#             mask[i, dropout_indices] = 0
#         return mask


# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True


# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# class GatherLayer(autograd.Function):
#     """
#     Gather tensors from all workers with support for backward propagation:
#     This implementation does not cut the gradients as torch.distributed.all_gather does.
#     """
#
#     @staticmethod
#     def forward(ctx, x):
#         output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
#         dist.all_gather(output, x)
#         return tuple(output)
#
#     @staticmethod
#     def backward(ctx, *grads):
#         all_gradients = torch.stack(grads)
#         dist.all_reduce(all_gradients)
#         return all_gradients[dist.get_rank()]


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()


# def all_gather_batch_with_grad(tensors):
#     """
#     Performs all_gather operation on the provided tensors.
#     Graph remains connected for backward grad computation.
#     """
#     # Queue the gathered tensors
#     world_size = get_world_size()
#     # There is no need for reduction in the single-proc case
#     if world_size == 1:
#         return tensors
#     tensor_list = []
#     output_tensor = []
#
#     for tensor in tensors:
#         tensor_all = GatherLayer.apply(tensor)
#         tensor_list.append(tensor_all)
#
#     for tensor_all in tensor_list:
#         output_tensor.append(torch.cat(tensor_all, dim=0))
#     return output_tensor


# class IntraLoss(nn.Module):
#
#     def __init__(self, temperature=0.1):
#         super().__init__()
#         self.tau = temperature
#         self.labels = None
#         self.masks = None
#         self.last_local_batch_size = None
#
#     def forward(self, q_a, q_b):
#         local_batch_size = q_a.size(0)
#
#         k_a, k_b = all_gather_batch_with_grad([q_a, q_b])
#
#         if local_batch_size != self.last_local_batch_size:
#             self.labels = local_batch_size * get_rank() + torch.arange(
#                 local_batch_size, device=q_a.device
#             )
#             total_batch_size = local_batch_size * get_world_size()
#             self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
#             self.last_local_batch_size = local_batch_size
#
#
#         logits_ab = 100 * q_a @ q_b.t()
#         logits_ba = logits_ab.t()
#
#
#         loss_ab = F.cross_entropy(logits_ab, self.labels)
#         loss_ba = F.cross_entropy(logits_ba, self.labels)
#         loss = (loss_ab + loss_ba) / 2  # divide by 2 to average over all samples
#
#         return loss
#
#
# class InterLoss(nn.Module):
#
#     def __init__(self, temperature=0.1):
#         super().__init__()
#         self.tau = temperature
#         self.labels = None
#         self.masks = None
#         self.last_local_batch_size = None
#
#     def forward(self, q_a, q_b):
#         local_batch_size = q_a.size(0)
#
#         if local_batch_size != self.last_local_batch_size:
#             self.labels = local_batch_size * get_rank() + torch.arange(
#                 local_batch_size, device=q_a.device
#             )
#             total_batch_size = local_batch_size * get_world_size()
#             self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
#             self.last_local_batch_size = local_batch_size
#         logits_aa = 100 * q_a @ q_b.t()
#
#         loss = F.cross_entropy(logits_aa, self.labels)
#
#         return loss



class UnifiedPrompt_vision(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        # Define the layers as before
        self.self_attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            ) for _ in range(num_layers)]
        )

        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, U, domain_vision_deep):

        # shallow
        combined_tensor = torch.cat([u.data.unsqueeze(0).cuda() for u in U], dim=0)


        attn_output, _ = self.self_attn_layers[0](combined_tensor, combined_tensor, combined_tensor)
        combined_tensor = combined_tensor + attn_output
        combined_tensor = self.ln1(combined_tensor)

        ffn_output = self.ffn_layers[0](combined_tensor)
        combined_tensor = combined_tensor + ffn_output
        combined_tensor = self.ln2(combined_tensor)

        vis_prompt_shallow = combined_tensor.mean(dim=0)

        # deep


        depth = len(domain_vision_deep[0])
        compound_prompts_vision = []
        for d in range(depth):
            inputs_vison = []

            for client in range(len(domain_vision_deep)):
                dlayer_tensor_vision = domain_vision_deep[client][d]
                inputs_vison.append(dlayer_tensor_vision)

            combined_tensor_vision = torch.cat([u.data.unsqueeze(0).cuda() for u in inputs_vison], dim=0)

            attn_output_vision, _ = self.self_attn_layers[d + 1](combined_tensor_vision, combined_tensor_vision,
                                                                 combined_tensor_vision)
            combined_tensor_vision = combined_tensor_vision + attn_output_vision
            combined_tensor_vision = self.ln1(combined_tensor_vision)

            ffn_output_vision = self.ffn_layers[d + 1](combined_tensor_vision)
            combined_tensor_vision = combined_tensor_vision + ffn_output_vision
            combined_tensor_vision = self.ln2(combined_tensor_vision)

            vis_prompt_deep_l = combined_tensor_vision.mean(dim=0)
            compound_prompts_vision.append(vis_prompt_deep_l)

        return vis_prompt_shallow, compound_prompts_vision


class UnifiedPrompt_text(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        # Define the layers as before
        self.self_attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            ) for _ in range(num_layers)]
        )

        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, U, domain_text_deep):

        # shallow
        combined_tensor = torch.cat([u.data.unsqueeze(0).cuda() for u in U], dim=0)
        combined_tensor = combined_tensor.float()

        attn_output, _ = self.self_attn_layers[0](combined_tensor, combined_tensor, combined_tensor)
        combined_tensor = combined_tensor + attn_output
        combined_tensor = self.ln1(combined_tensor)

        ffn_output = self.ffn_layers[0](combined_tensor)
        combined_tensor = combined_tensor + ffn_output
        combined_tensor = self.ln2(combined_tensor)

        vis_prompt_shallow = combined_tensor.mean(dim=0)

        # deep

        depth = len(domain_text_deep[0])
        compound_prompts_text = []
        for d in range(depth):
            inputs_vison = []

            for client in range(len(domain_text_deep)):
                dlayer_tensor_text = domain_text_deep[client][d]
                inputs_vison.append(dlayer_tensor_text)

            combined_tensor_text = torch.cat([u.data.unsqueeze(0).cuda() for u in inputs_vison], dim=0)
            combined_tensor_text = combined_tensor_text

            attn_output_text, _ = self.self_attn_layers[d + 1](combined_tensor_text, combined_tensor_text,
                                                               combined_tensor_text)
            combined_tensor_text = combined_tensor_text + attn_output_text
            combined_tensor_text = self.ln1(combined_tensor_text)

            ffn_output_text = self.ffn_layers[d + 1](combined_tensor_text)
            combined_tensor_text = combined_tensor_text + ffn_output_text
            combined_tensor_text = self.ln2(combined_tensor_text)

            vis_prompt_deep_l = combined_tensor_text.mean(dim=0)
            compound_prompts_text.append(vis_prompt_deep_l)

        return vis_prompt_shallow, compound_prompts_text

# def load_clip_to_cpu(args):
#     backbone_name = args.backbone_name
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url, args.root_dir)
#
#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None
#
#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")
#     design_details = {"trainer": 'MaPLe',
#                       "vision_depth": 0,
#                       "language_depth": 0, "vision_ctx": 0,
#                       "language_ctx": 0,
#                       "maple_length": args.N_CTX}
#     model = clip.build_model(state_dict or model.state_dict(), design_details)
#
#     return model

def load_clip_to_cpu(args,zero_shot = False):
    backbone_name = args.backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, args.root_dir)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot:

        design_details = {"trainer": 'MaPLe',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0,
                          "maple_length": args.N_CTX}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}

        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.N_CTX  # 2
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.INPUT_SIZE[0]
        ctx_init = args.CTX_INIT
        # Default is 1, which is compound shallow prompting
        assert args.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = args.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if ctx_init and (n_ctx) <= 4:
        #     # use given words to initialize context vectors
        #     ctx_init = ctx_init.replace("_", " ")
        #     n_ctx = n_ctx
        #     prompt = clip.tokenize(ctx_init)
        #     with torch.no_grad():
        #         embedding = clip_model.token_embedding(prompt).type(dtype)
        #     ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        #     prompt_prefix = ctx_init
        # else:
        #     random initialization

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        ##########
        # random initiali
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.VPTctxList = nn.ParameterList([nn.Parameter(ctx_vectors)
                                            for _ in range(5)])

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)



        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.VPTList = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                         for _ in range(5)])
        for single_para in self.VPTList:
            nn.init.normal_(single_para, std=0.02)

        # self.compound_prompts_vision = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
        #                                                for _ in range(self.compound_prompts_depth - 1)])

        self.VPTcompound_prompts_vision = nn.ParameterList()
        for _ in range(5):
            client_i_deep = nn.ParameterList([nn.Parameter(torch.empty(args.N_CTX, 768))
                                              for _ in range(args.PROMPT_DEPTH - 1)])
            self.VPTcompound_prompts_vision.append(client_i_deep)

        for single_para in self.VPTcompound_prompts_vision:
            for _ in single_para:
                nn.init.normal_(_, std=0.02)

        self.VPTcompound_prompts_text = nn.ParameterList()
        for _ in range(5):
            client_i_deep = nn.ParameterList([nn.Parameter(torch.empty(args.N_CTX, 512))
                                              for _ in range(args.PROMPT_DEPTH - 1)])

            self.VPTcompound_prompts_text.append(client_i_deep)

        for single_para in self.VPTcompound_prompts_text:
            for _ in single_para:
                nn.init.normal_(_, std=0.02)

        self.global_compound_prompts_vision = []
        for _ in range(5):
            client_i_deep = [torch.empty(args.N_CTX_VISION, 768, requires_grad=False)
                             for _ in range(args.PROMPT_DEPTH - 1)]
            self.global_compound_prompts_vision.append(client_i_deep)

        self.global_compound_prompts_text = []
        for _ in range(5):
            client_i_deep = [torch.empty(args.N_CTX_TEXT, 512, requires_grad=False)
                             for _ in range(args.PROMPT_DEPTH - 1)]
            self.global_compound_prompts_text.append(client_i_deep)

        self.global_ctxList = [torch.empty(n_ctx, 512, requires_grad=False)
                               for _ in range(5)]

        self.global_List = [torch.empty(n_ctx, 768, requires_grad=False)
                            for _ in range(5)]

        clip_model_temp_image = load_clip_to_cpu(args, True)
        self.ZS_image_encoder = clip_model_temp_image.visual


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.communication_i = 0
        self.client_i = 0
        self.test_envs = int(args.test_envs[0])
        self.depth_vision = args.PROMPT_DEPTH

        self.depth_text = args.PROMPT_DEPTH
        self.if_loss_domain = args.if_loss_domain
        self.N_CTX_TEXT = args.N_CTX_TEXT
        self.PROMPT_DEPTH = args.PROMPT_DEPTH
        # self.PROMPT_DEPTH_TEXT = args.PROMPT_DEPTH

        self.UnifiedPrompt_vision = UnifiedPrompt_vision(embed_dim=768, num_heads=8, ff_dim=1536,
                                                         num_layers=self.PROMPT_DEPTH)
        self.UnifiedPrompt_text = UnifiedPrompt_text(embed_dim=512, num_heads=8, ff_dim=1024,
                                                     num_layers=self.PROMPT_DEPTH)
        self.batch_id = 0
        self.a_t = args.a_t
        self.b_t = args.b_t
        self.c = args.c
        self.d =args.d
        self.e =args.e
        self.T = 1.0
        self.lambada_ = args.lambada_
        self.ZS_image_encoder = self.prompt_learner.ZS_image_encoder
        self.fixed_embeddings = 0


    def get_global_prompts(self):

        VPTcompound_prompts_vision = self.prompt_learner.VPTcompound_prompts_vision
        VPTcompound_prompts_text = self.prompt_learner.VPTcompound_prompts_text
        VPTctxList = self.prompt_learner.VPTctxList
        VPTList = self.prompt_learner.VPTList


        self.prompt_learner.global_compound_prompts_vision = [[param.clone() for param in client_i] for client_i in
                                           VPTcompound_prompts_vision]

        self.prompt_learner.global_compound_prompts_text = [[param.clone() for param in client_i] for client_i in
                                         VPTcompound_prompts_text]

        self.prompt_learner.global_ctxList = [param.clone() for param in VPTctxList]
        self.prompt_learner.global_List = [param.clone() for param in VPTList]



    def forward(self, image, label=None, training=False):

        c = self.c

        # text_loss = self.text_loss
        # vis_loss = self.vis_loss


        UnifiedPrompt_vision = self.UnifiedPrompt_vision
        UnifiedPrompt_text = self.UnifiedPrompt_text

        VPTlist = self.prompt_learner.VPTList
        VPT_ctx_list = self.prompt_learner.VPTctxList

        client_i = self.client_i
        communication = self.communication_i
        test_envs = self.test_envs
        if client_i - test_envs < 0:
            VPT_i = client_i
        else:
            VPT_i = client_i - 1

        VPT_shallow = VPTlist[VPT_i]


        VPTcompound_prompts_text = self.prompt_learner.VPTcompound_prompts_text
        VPTcompound_prompts_vision = self.prompt_learner.VPTcompound_prompts_vision

        deep_compound_prompts_text_list = self.prompt_learner.VPTcompound_prompts_text[VPT_i]
        deep_compound_prompts_vision_list = self.prompt_learner.VPTcompound_prompts_vision[VPT_i]

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prefix = self.prompt_learner.token_prefix
        suffix = self.prompt_learner.token_suffix

        if training:

            if communication %2 != 0:
                for name, param in UnifiedPrompt_text.named_parameters():
                    param.requires_grad_(True)
                for name, param in UnifiedPrompt_text.named_parameters():
                    param.requires_grad_(True)

                for VPT in VPT_ctx_list:
                    VPT.requires_grad_(False)

                for VPT in VPT_ctx_list:
                    VPT.requires_grad_(False)

                for param_list in self.prompt_learner.VPTcompound_prompts_vision:
                    for param in param_list:
                        param.requires_grad_(False)

                for param_list in self.prompt_learner.VPTcompound_prompts_text:
                    for param in param_list:
                        param.requires_grad_(False)

                adapter_prompt_vison_shallow, update_VPT_deep = UnifiedPrompt_vision(
                    VPTlist, VPTcompound_prompts_vision)

                adapter_prompt_text_shallow, update_text_deep = UnifiedPrompt_text(
                    VPT_ctx_list, VPTcompound_prompts_text)

                if adapter_prompt_text_shallow.dim() == 2:
                    adapter_prompt_text_shallow = adapter_prompt_text_shallow.unsqueeze(0).expand(
                        self.prompt_learner.n_cls, -1, -1)


                prompts_adapter = self.prompt_learner.construct_prompts(adapter_prompt_text_shallow, prefix, suffix)

                image_features_ap = self.image_encoder(image.type(self.dtype), adapter_prompt_vison_shallow,
                                                       update_VPT_deep)
                image_features_ap = image_features_ap / image_features_ap.norm(dim=-1, keepdim=True)

                prompts_adapter = prompts_adapter.half()

                text_features_ap = self.text_encoder(prompts_adapter, tokenized_prompts, update_text_deep)
                text_features_ap = text_features_ap / text_features_ap.norm(dim=-1, keepdim=True)

                logits_ada = logit_scale * image_features_ap @ text_features_ap.t()

                loss_ada = F.cross_entropy(logits_ada, label)

                return loss_ada

            else:
                for index_VPT, VPT in enumerate(VPTlist):
                    if index_VPT==VPT_i:
                        VPT.requires_grad_(True)
                    else:
                        VPT.requires_grad_(False)
                for index_VPT, VPT in enumerate(VPT_ctx_list):
                    if index_VPT == VPT_i:
                        VPT.requires_grad_(True)
                    else:
                        VPT.requires_grad_(False)

                for index_VPTlist,param_list in enumerate(self.prompt_learner.VPTcompound_prompts_vision):
                    if index_VPTlist == VPT_i:
                        for param in param_list:
                            param.requires_grad_(True)
                    else:
                        for param in param_list:
                            param.requires_grad_(False)

                for index_VPTlist,param_list in enumerate(self.prompt_learner.VPTcompound_prompts_text):
                    if index_VPTlist == VPT_i:
                        for param in param_list:
                            param.requires_grad_(True)
                    else:
                        for param in param_list:
                            param.requires_grad_(False)

                for name, param in UnifiedPrompt_text.named_parameters():
                    param.requires_grad_(False)
                for name, param in UnifiedPrompt_text.named_parameters():
                    param.requires_grad_(False)


                if communication == 0:
                    fixed_embeddings = self.fixed_embeddings.cuda()  # precomputed pre-trained frozen textual features
                    fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)

                    with torch.no_grad():
                        zero_shot_features = self.ZS_image_encoder(image.type(self.dtype))
                        zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                        # Compute pre-trained frozen visual features
                        tea_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()

                    ctx = VPT_ctx_list[VPT_i]
                    if ctx.dim() == 2:
                        ctx1 = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)

                    prompts = self.prompt_learner.construct_prompts(ctx1, prefix, suffix).half()

                    text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text_list)
                    image_features = self.image_encoder(image.type(self.dtype), VPT_shallow,
                                                        deep_compound_prompts_vision_list)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    logits = logit_scale * image_features @ text_features.t()

                    tea_prob = F.softmax(tea_logits / self.T, dim=-1)
                    kl_loss = -tea_prob * F.log_softmax(logits / self.T,
                                                        -1) * self.T * self.T
                    kl_loss = kl_loss.sum(1).mean()

                    loss = F.cross_entropy(logits, label)
                    return loss + c * kl_loss
                else:
                    #global prompt
                    adapter_prompt_vison_shallow_pre, update_VPT_deep_pre = UnifiedPrompt_vision(
                        self.prompt_learner.global_List, self.prompt_learner.global_compound_prompts_vision)

                    adapter_prompt_text_shallow_pre, update_text_deep_pre = UnifiedPrompt_text(
                        self.prompt_learner.global_ctxList, self.prompt_learner.global_compound_prompts_text)

                    #current
                    adapter_prompt_vison_shallow, update_VPT_deep = UnifiedPrompt_vision(
                        self.prompt_learner.VPTList, self.prompt_learner.VPTcompound_prompts_vision)

                    adapter_prompt_text_shallow, update_text_deep = UnifiedPrompt_text(
                        self.prompt_learner.VPTctxList, self.prompt_learner.VPTcompound_prompts_text)


                    if adapter_prompt_text_shallow_pre.dim() == 2:
                        adapter_prompt_text_shallow_pre = adapter_prompt_text_shallow_pre.unsqueeze(0).expand(
                            self.prompt_learner.n_cls, -1, -1)
                    #current
                    if adapter_prompt_text_shallow.dim() == 2:
                        adapter_prompt_text_shallow_cur = adapter_prompt_text_shallow.unsqueeze(0).expand(
                            self.prompt_learner.n_cls, -1, -1)


                    prefix = self.prompt_learner.token_prefix
                    suffix = self.prompt_learner.token_suffix

                    # global features
                    prompts_adapter = self.prompt_learner.construct_prompts(adapter_prompt_text_shallow_pre, prefix,
                                                                            suffix)
                    image_features_ap = self.image_encoder(image.type(self.dtype), adapter_prompt_vison_shallow_pre,
                                                           update_VPT_deep_pre)
                    image_features_ap = image_features_ap / image_features_ap.norm(dim=-1, keepdim=True)

                    prompts_adapter = prompts_adapter.half()

                    text_features_ap = self.text_encoder(prompts_adapter, tokenized_prompts, update_text_deep_pre)
                    text_features_ap = text_features_ap / text_features_ap.norm(dim=-1, keepdim=True)

                    # current
                    prompts_adapter_cur = self.prompt_learner.construct_prompts(adapter_prompt_text_shallow_cur, prefix,
                                                                            suffix)

                    image_features_cur = self.image_encoder(image.type(self.dtype), adapter_prompt_vison_shallow,
                                                           update_VPT_deep)

                    image_features_cur = image_features_cur / image_features_cur.norm(dim=-1, keepdim=True)

                    prompts_adapter_cur = prompts_adapter_cur.half()

                    text_features_cur = self.text_encoder(prompts_adapter_cur, tokenized_prompts, update_text_deep)
                    text_features_cur = text_features_cur / text_features_cur.norm(dim=-1, keepdim=True)

                    logits = logit_scale * image_features_cur @ text_features_cur.t()
                    loss = F.cross_entropy(logits, label)

                    tea_logits = logit_scale * image_features_ap @ text_features_ap.t()
                    tea_prob = F.softmax(tea_logits / self.T, dim=-1)
                    kl_loss = -tea_prob * F.log_softmax(logits / self.T,
                                                        -1) * self.T * self.T
                    kl_loss = kl_loss.sum(1).mean()

                    return loss + c * kl_loss

        else:


            adapter_prompt_vison_shallow, update_VPT_deep = UnifiedPrompt_vision(
                VPTlist, VPTcompound_prompts_vision)

            adapter_prompt_text_shallow, update_text_deep = UnifiedPrompt_text(
                VPT_ctx_list, VPTcompound_prompts_text)

            if adapter_prompt_text_shallow.dim() == 2:
                adapter_prompt_text_shallow = adapter_prompt_text_shallow.unsqueeze(0).expand(
                    self.prompt_learner.n_cls, -1, -1)

            prefix = self.prompt_learner.token_prefix
            suffix = self.prompt_learner.token_suffix

            prompts_adapter = self.prompt_learner.construct_prompts(adapter_prompt_text_shallow, prefix, suffix)
            image_features_ap = self.image_encoder(image.type(self.dtype), adapter_prompt_vison_shallow,
                                                   update_VPT_deep)
            image_features_ap = image_features_ap / image_features_ap.norm(dim=-1, keepdim=True)
            prompts_adapter = prompts_adapter.half()

            text_features_ap = self.text_encoder(prompts_adapter, tokenized_prompts, update_text_deep)
            text_features_ap = text_features_ap / text_features_ap.norm(dim=-1, keepdim=True)

            logits_ada = logit_scale * image_features_ap @ text_features_ap.t()


            return logits_ada,image_features_ap


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
