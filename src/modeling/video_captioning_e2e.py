from fairscale.nn.misc import checkpoint_wrapper
import torch


class SimpleRMSNorm(torch.nn.Module):

    def __init__(self, d, p=-1., eps=1e-6, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(SimpleRMSNorm, self).__init__()
        self.eps = eps
        self.d = d

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x**(-1. / 2)
        x_normed = x / (rms_x + self.eps)

        return x_normed


class VideoTransformer(torch.nn.Module):

    def __init__(self, args, config, transformer_encoder):
        super(VideoTransformer, self).__init__()
        self.config = config
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = 1024
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.compute_mask_on_the_fly = False  # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard',
                                             False)


        self.norm = SimpleRMSNorm(d=self.img_feature_dim)

        if self.learn_mask_enabled == True:
            self.learn_vid_att = torch.nn.Embedding(
                args.max_img_seq_length * args.max_img_seq_length, 1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):

        # prepare VL transformer inputs
        #kwargs['img_feats'] = vid_feats

        # self.trans_encoder.bert.encoder.output_attention = False
        # if self.trans_encoder.bert.encoder.output_attentions:
        #     self.trans_encoder.bert.encoder.set_output_attentions(False)

        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            # max_img_seq_length = 784
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(
                vid_att_len, vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask) * learn_att
            learn_att = diag_mask + video_attention
            # sparse_mask_soft2hard = False
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att >= 0.5) * 1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::,
                                     -vid_att_len::] = learn_att

        outputs = self.trans_encoder(*args, **kwargs)
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)
            outputs = outputs + (loss_sparsity, )

        return outputs  # outputs = (loss= NUM, logits= torch.Size([123, 30522]))

    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length / pretrained_num_tokens

        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens *
                          (i + 1),
                          pretrained_num_tokens * i:pretrained_num_tokens *
                          (i + 1)] = pretrained_learn_att

    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        scale_factor = int(self.max_img_seq_length / pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(
                pretrained_learn_att[None,
                                     None, :, :].double())[0, 0, :, :].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(
            self.max_img_seq_length * self.max_img_seq_length, 1)

    def reload_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens *
                          (i + 1),
                          pretrained_num_tokens * i:pretrained_num_tokens *
                          (i + 1)] = pretrained_learn_att