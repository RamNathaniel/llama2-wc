import json
import torch

from statistics import mean
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer
from typing import Tuple, List

from phraser.phraser import Loopback
from utils.llama_utils import LlamaUtils


class PhraserUtils:
    MODELS_FOLDER = f'{LlamaUtils.HOME}/phraser-models'

    LAYER_TO_USE = 30

    @staticmethod
    def run_llama_on_tokens(model: Transformer, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens_tensor = torch.unsqueeze(torch.tensor(tokens).long(), 0).cuda()

            model.training = False
            model.layer_output_ind = PhraserUtils.LAYER_TO_USE
            logits = model(tokens_tensor, 0)
            probs = torch.nn.functional.softmax(logits, dim=1)[0, :]
            idea = model.layer_output

        return probs, idea

    @staticmethod
    def run_llama_on_batch(model: Transformer, tokens: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens_tensor = torch.tensor(tokens).long().cuda()

            model.training = False
            model.layer_output_ind = PhraserUtils.LAYER_TO_USE
            logits = model(tokens_tensor, 0)
            probs = torch.nn.functional.softmax(logits, dim=1)
            idea = model.layer_output
        
        return probs, idea


    @staticmethod
    def create_params() -> ModelArgs:
        with open(Path(LlamaUtils.CHECKPOINT_DIR) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=LlamaUtils.MAX_SEQ_LEN,
            max_batch_size=LlamaUtils.MAX_BATCH_SIZE,
            **params
        )

        model_args.n_layers = model_args.n_layers - PhraserUtils.LAYER_TO_USE
        return model_args


    @staticmethod
    def clone_initial_phraser(target_path: str=MODELS_FOLDER+'/phraser.pth'):
        llama_model, tokenizer = LlamaUtils.load_model()
        llama_model.layers = llama_model.layers[PhraserUtils.LAYER_TO_USE-1:]
        llama_model.tok_embeddings = None

        torch.save(llama_model.state_dict(), target_path)
        pass

    @staticmethod
    def run_phraser_on_tokens(
            tokens: List[int],
            llama_model: Transformer,
            phraser: Transformer,
            loopback: Loopback) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run the Llama+Phraser loop on a list of tokens.
        # Check the generation file for how to stop the loop.
        with torch.no_grad():
            tokens_tensor = torch.unsqueeze(torch.tensor(tokens).long(), 0).cuda()

            llama_model.training = False
            llama_model.layer_output_ind = PhraserUtils.LAYER_TO_USE
            _logits = llama_model(tokens_tensor, 0)  # before softmax
            idea = llama_model.layer_output

            done = False
            while not done:
                phraser.training = False
                probs = phraser(idea)
                probs = torch.nn.functional.softmax(probs, dim=1)[0, :]

                # Get the next token
                next_token = torch.argmax(probs).item()

                # Update the tokens_tensor
                idea = loopback(idea, next_token)

        return probs, idea
