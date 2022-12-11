from Precomputing.Ensembling import EnGCN
from trainer import load_data
from options.base_options import BaseOptions
import torch

def main(args):
    (
        data,
        x,
        y,
        split_masks,
        evaluator,
        processed_dir,
    ) = load_data(args.dataset, args.tosparse)

    # Create enGCN model
    en_gcn = EnGCN(args, data, evaluator)
    # Load model values
    en_gcn.model.load_state_dict(
        torch.load(f"{en_gcn.type_model}_{en_gcn.dataset}_MLP_SLE.pt")
    )

    device = torch.device(f"cuda:{args.cuda_num}" if args.cuda else "cpu")
    input_dict = {
        "split_masks": split_masks,
        "data": data,
        "x": x,
        "y": y,
        "device": device,
    }
    en_gcn.inference(input_dict)

# Call with same params as used for training
if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)
