"""Smoke test: verify thumbnail-based architecture works on GPU."""
import torch
from adatile.config import Config
from adatile.modeling import build_adatile_fastsam

cfg = Config()
cfg.backbone.name = 'ResNet50Backbone'
cfg.backbone.pretrained = True
cfg.sparse.name = 'ada_spm'
cfg.sparse.num_scales = 4
cfg.tokenizer.name = 'dynamic_tile'
cfg.tokenizer.tile_sizes = [384, 768, 1536]
cfg.tokenizer.max_tokens_per_image = 128
cfg.tokenizer.skip_mode = 'threshold'
cfg.router.name = 'DTRv2Router'
cfg.router.embed_dim = 256
cfg.decoder.name = 'fastsam_decoder'
cfg.prototype.name = ''

print('Building model...')
model = build_adatile_fastsam(cfg)
model = model.cuda()
model.eval()

print('Running forward pass on 2048×2048...')
image = torch.randn(1, 3, 2048, 2048).cuda()
with torch.no_grad():
    output, aux = model(image)

print(f'  Masks: {output.masks.shape}')
print(f'  Scores: {output.scores.shape}')
print(f'  Importance: {aux["importance"].shape}')
print(f'  Peak GPU: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
print('DONE — architecture works!')
