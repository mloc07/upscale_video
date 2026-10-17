import torch
from rotary_embedding_torch import RotaryEmbedding

rope = RotaryEmbedding(dim=42, freqs_for="lang", theta=10000)

class DummyNaMM:
    def __init__(self, rope):
        self.rope = rope

    def get_axial_freqs(self, *dims):
        return self.rope.get_axial_freqs(*dims)

    def original_get_freqs(self, vid_shape, txt_shape):
        vid_freqs = self.get_axial_freqs(1024, 128, 128)
        txt_freqs = self.get_axial_freqs(1024)
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq = vid_freqs[l : l + f, :h, :w].reshape(-1, vid_freqs.size(-1))
            txt_freq = txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1))
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)

    def optimized_get_freqs(self, vid_shape, txt_shape):
        vid_shapes = vid_shape.tolist()
        txt_lens = txt_shape[:, 0].tolist()

        max_f_idx = max(l + f for (f, h, w), l in zip(vid_shapes, txt_lens))
        max_h = max(h for f, h, w in vid_shapes)
        max_w = max(w for f, h, w in vid_shapes)
        max_l = max(txt_lens)

        device = self.rope.device
        pos0 = torch.arange(max_f_idx, device=device)
        pos1 = torch.arange(max_h, device=device)
        pos2 = torch.arange(max_w, device=device)

        freqs0 = self.rope.forward(pos0, seq_len=max_f_idx)
        freqs1 = self.rope.forward(pos1, seq_len=max_h)
        freqs2 = self.rope.forward(pos2, seq_len=max_w)

        d = freqs0.size(-1) * 3
        txt_freqs_all = freqs0[:max_l].repeat(1, 3).reshape(-1, d)

        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shapes, txt_lens):
            f0 = freqs0[l : l + f, None, None, :]
            f1 = freqs1[None, :h, None, :]
            f2 = freqs2[None, None, :w, :]
            f0_b, f1_b, f2_b = torch.broadcast_tensors(f0, f1, f2)
            vid_freq = torch.cat([f0_b, f1_b, f2_b], dim=-1).reshape(-1, d)
            txt_freq = txt_freqs_all[:l]
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)

dummy = DummyNaMM(rope)
vid_shape = torch.tensor([[4, 3, 3], [2, 3, 3]])
txt_shape = torch.tensor([[10], [10]])

orig_vid, orig_txt = dummy.original_get_freqs(vid_shape, txt_shape)
opt_vid, opt_txt = dummy.optimized_get_freqs(vid_shape, txt_shape)

diff_vid = (orig_vid - opt_vid).abs().max().item()
diff_txt = (orig_txt - opt_txt).abs().max().item()
print("diff_vid:", diff_vid, "diff_txt:", diff_txt)
assert diff_vid == 0.0 and diff_txt == 0.0, "MUST BE IDENTICAL!"
print("COMPLETE TEST SUCCESS! 100% BIT-EXACT!")
