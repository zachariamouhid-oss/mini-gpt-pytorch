import torch
from model import Transformer
from tokenizer import Tokenizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def generate_tokens(config_model_path: str, weight_path: str, config_tokenizer_path: str, input : str, max_new_tokens = 256, temperature = 1):

    model = Transformer.import_model(config_model_path, weight_path, device = device)
    model.to(device=device)
    tok = Tokenizer.from_file(config_tokenizer_path)

    idx_list = tok.encode(input)
    idx = torch.tensor(idx_list, device=device).unsqueeze(0)
    idx = model.generate(idx, max_new_tokens, temperature, tok.eos_token_id)

    output_text=tok.decode(idx[0].tolist())
    print(output_text)

    return output_text

if __name__ == "__main__":
    model_config = "model_config.json"
    current_model = "zachGPT_pretrained.pt" 
    config_tok = "tokenizer_wiki_16k.json"
    print(f"\nPrompt : \"La france est\"")
    generate_tokens(model_config, current_model, config_tok, "La france est", 256, 1)
    print(f"\nPrompt : \"Les mathématiques sont un ensemble de connaissances abstraites résultant de raisonnements logiques appliqués à des objets divers\"")
    generate_tokens(model_config, current_model, config_tok, "Les mathématiques sont un ensemble de connaissances abstraites résultant de raisonnements logiques appliqués à des objets divers", 256,1)
    print(f"\nPrompt : \"En 1928, le gouvernement arménien intensifia la collectivisation forcée\"")
    generate_tokens(model_config, current_model, config_tok, "En 1928, le gouvernement arménien intensifia la collectivisation forcée", 256, 1)
    