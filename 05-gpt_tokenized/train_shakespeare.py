import torch

from bpe_tokenizer import ENDOFTEXT_TOKEN, BpeTokenizer, BpeTrainer
from gpt_model import ModelConfig, GptLanguageModel
from data_provider import DataProvider
from gpt_trainer import GPTTrainer



def main():
    seed = 31337
    torch.manual_seed(seed)
    dataset_name = "shakespeare_complete"
    vocab_size = 1024
    special_tokens = [ENDOFTEXT_TOKEN]
    verbose = True
    n_layer = 6
    n_embd = 384 # must be: n_embd = num_heads * head_size
    num_heads = 12
    head_size = 32
    block_size = 512
    batch_size = 32
    dropout = 0.3
    lr = 0.0005
    
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    config = ModelConfig(verbose=verbose, n_layer=n_layer, num_heads=num_heads, n_embd=n_embd, 
                         batch_size=batch_size, block_size=block_size, head_size=head_size, dropout=dropout, 
                         lr=lr, device=device, special_tokens=special_tokens, split_tokens=special_tokens,
                         vocab_size=vocab_size,dataset_name=dataset_name, seed=seed)
    
    config.verbose = False
    bpeTrainer = BpeTrainer(config)
    vocab, merges, special_tokens = bpeTrainer.load_or_train()
    config.verbose = True
    tokenizer = BpeTokenizer(vocab, merges, special_tokens, verbose)
    provider = DataProvider(config, tokenizer)
    model = GptLanguageModel(config)
    # model = torch.compile(model)

    trainer = GPTTrainer(config, model, tokenizer, provider)
    trainer.train_weights(8000)

    prompt_text = f"{ENDOFTEXT_TOKEN}"
    prompt = tokenizer.encode(prompt_text)

    response = model.generate(prompt, 1024)
    response_text = tokenizer.decode(response.tolist())
    print(response_text)
if __name__ == "__main__":
    main()