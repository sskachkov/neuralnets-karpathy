import time
import torch
import math 

from gpt_model import ModelConfig


class GPTTrainer:
    def __init__(self, config:ModelConfig, model, tokenizer, provider):
        self.config = config
        self.tokenizer = tokenizer
        self.provider = provider
        self.model = model
        self.optimizer = None
        self.use_amp = self.config.device == 'mps'

        if self.config.verbose:
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"GPTTrainer created for dataset: {self.config.dataset_name}, device: {self.config.device}, "
                  f"chars per token: {self.provider.chars_per_token:.2f}, params: {param_count/1e6:.2f}M, \nconfig: {self.config}")

    def train_weights(self, iters):
        provider = self.provider
        if self.config.verbose:
            print(f"Starting training for {iters} iterations, train dataset size: {len(provider.get_data('train'))/1e6:.2f}M, val dataset size: {len(provider.get_data('val'))/1e6:.2f}M")
        self.model.train()

        decay = [p for p in self.model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': 0.1},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=self.config.lr)

        warmup = max(1, iters // 50)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=iters - warmup, eta_min=self.config.lr * 0.1),
            ],
            milestones=[warmup],
        )

        start_time = time.time()
        prev_time = start_time
        for i in range(iters):
            if i % (iters // 10) == 0 or i == iters - 1:
                tl, vl= self._estimate_loss()
                cpt = provider.chars_per_token
                tlbpc = tl / cpt / math.log(2)
                vlbpc = vl / cpt / math.log(2)
                
                curr_time = time.time()
                print(f"Iter: {i:4d}, lr: {float(self.scheduler.get_last_lr()[0]):8f}, train loss: {tl:.4f} ({tlbpc:.4f} bpc), val loss: {vl:.4f} ({vlbpc:.4f} bpc), elapsed: {int(curr_time - prev_time)}s ({int(curr_time - start_time)} total).")
                prev_time = curr_time
                self.model.train()

            xb, yb = provider.get_batch('train')
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16, enabled=self.use_amp):
                logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)            
            self.optimizer.step()
            self.scheduler.step()


    @torch.no_grad
    def _estimate_loss(self):
        self.model.eval()
        iters = 100
        train_loss, val_loss = 0, 0
        with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16, enabled=self.use_amp):
            for _ in range(iters):
                xb, yb = self.provider.get_batch('train')
                _, loss = self.model(xb, yb)
                train_loss += loss
                xb, yb = self.provider.get_batch('val')
                _, loss = self.model(xb, yb)
                val_loss += loss
        train_loss /= iters
        val_loss /= iters
        return train_loss.item(), val_loss.item()

