import gc
from default_paths import PATH_TO_SIMCLR_IMAGENET
import torch

from collections import defaultdict
from pathlib import Path
import pickle

from classification.classification_module import ClassificationModule
from tqdm.autonotebook import tqdm

from classification.vit_models import mae_vit_base_patch16
from torchvision.transforms.functional import center_crop


def get_ids_from_model_names(encoder_name, model_name):
    encoder_id = (
        Path(encoder_name).parent.stem[4:]
        if (
            "imagenet" not in encoder_name
            and encoder_name not in ["raddino", "cxr_mae", "imagenet_mae", "random"]
        )
        else encoder_name
    )

    if model_name is None:
        model_id = "no_model"
    else:
        model_id = Path(model_name).parent.stem[4:]

    return encoder_id, model_id


def get_or_save_outputs(
    model_to_evaluate, 
    encoder_to_evaluate, 
    val_loader, 
    test_loader, 
    dataset_name,
    feat_mode: str = "all",  # options: "final", "early", "all"
):
    """
    Inference loop. If already saved simply returns dictionary of outputs.
    Else computes results for task model and encoder model, saves and returns the results.
    """
    encoder_id, model_id = get_ids_from_model_names(
        encoder_to_evaluate, model_to_evaluate
    )
    outputs_dir = Path(f"outputs/{dataset_name}")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_filename = outputs_dir / f"model_{model_id}.pkl"
    print(encoder_id)
    encoder_filename = outputs_dir / f"encoder_{encoder_id}.pkl"
    print(model_filename, encoder_filename)
    compute_task = model_to_evaluate is not None
    compute_encoder = True

    if model_filename.exists():
        with open(str(model_filename), "rb") as fp:
            task_output = pickle.load(fp)
            compute_task = model_to_evaluate is not None
    else:
        task_output = {}

    if model_to_evaluate is not None:
        model = ClassificationModule.load_from_checkpoint(
            model_to_evaluate, map_location="cuda:0", strict=False
        ).model.eval()
        model.cuda()
    else:
        compute_task = False

    if encoder_filename.exists():
        with open(str(encoder_filename), "rb") as fp:
            encoder_output = pickle.load(fp)
            compute_encoder = False
    else:
        encoder_output = {}
        match encoder_to_evaluate:
            case "imagenet":
                encoder = ClassificationModule(
                    num_classes=2,
                    encoder_name="resnet50",
                    input_channels=1 if dataset_name != "Retina" else 3,
                    pretrained=True,
                ).model.eval()
            case "simclr_imagenet":
                # From https://github.com/AndrewAtanov/simclr-pytorch/blob/master/README.md
                model_weights = PATH_TO_SIMCLR_IMAGENET  # noqa
                # Converting state dict to my model wrapper
                state_dict = torch.load(model_weights)["state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if "fc" not in k and "projection" not in k:
                        new_state_dict[k.replace("convnet.", "model.net.")] = v
                encoder_module = ClassificationModule(
                    num_classes=2, encoder_name="resnet50", input_channels=3
                )
                encoder_module.load_state_dict(new_state_dict, strict=False)
                encoder = encoder_module.model.eval()
            case "retfound":
                encoder = ClassificationModule(
                    num_classes=2, encoder_name="retfound", input_channels=3
                ).model.eval()
            case "cxr_mae" | "embed_mae":
                encoder = ClassificationModule(
                    num_classes=2, encoder_name=encoder_to_evaluate, input_channels=1
                ).model.eval()
            case "imagenet_mae":
                encoder = mae_vit_base_patch16(img_size=224, in_chans=3)
                encoder.load_state_dict(
                    torch.load("mae_pretrain_vit_base.pth")["model"], strict=False
                )
                encoder.eval()
            case "random":
                encoder = ClassificationModule(
                    num_classes=2,
                    encoder_name="resnet50",
                    pretrained=False,
                    input_channels=1 if dataset_name.lower() != "retina" else 3,
                ).model.eval()
            case _:
                try:
                    encoder = ClassificationModule.load_from_checkpoint(
                        encoder_to_evaluate,
                        map_location="cuda:0",
                        strict=False,
                        encoder_name="resnet50",
                    ).model.eval()
                except RuntimeError:
                    encoder = ClassificationModule.load_from_checkpoint(
                        encoder_to_evaluate,
                        map_location="cuda:0",
                        strict=False,
                        encoder_name="resnet18",
                    ).model.eval()
        encoder = encoder.to("cuda")

    if compute_task or compute_encoder:
        for name, loader in [("val", val_loader), ("test", test_loader)]:
            print(f"Processing {name} set...")
            y_list = []
            probas = []
            encoder_feats = []
            encoder_early_feats = []
            all_features = defaultdict(list)

            with torch.no_grad():
                for batch in tqdm(loader):
                    x = batch["x"].cuda()
                    y = batch["y"]
                    y_list.append(y)
                    if compute_task:
                        probas.append(model(x).cpu())
                    if compute_encoder:
                        # EMBED data by default 224*192. Not compatible with imagenet mae resize to 224*224
                        if encoder_to_evaluate == "imagenet_mae" and x.shape[-1] != 224:
                            x = center_crop(x, 224)
                        # To handle 1-channel images for encoders pretrained
                        # with RGB images (e.g. SimCLR ImageNet):
                        if encoder.input_channels == 3 and x.shape[1] == 1:
                            x = torch.repeat_interleave(x, 3, 1)
                        try:
                            if feat_mode == "all":
                                all_layer_feats = encoder.get_features(x, return_all_layers=True)
                                for layer_name, feat_tensor in all_layer_feats.items():
                                    all_features[layer_name].append(feat_tensor.detach().cpu())
                            elif feat_mode == "early":
                                early_feat, final_feat = encoder.get_features(x, include_early_feats=True)
                                encoder_early_feats.append(early_feat.detach().cpu())
                                encoder_feats.append(final_feat.detach().cpu())
                            else:  # "final"
                                final_feat = encoder.get_features(x)
                                encoder_feats.append(final_feat.detach().cpu())

                        except TypeError as e:
                            print(f"[Warning] Encoder get_features() failed: {e}")
                            final_feat = encoder.get_features(x)
                            encoder_feats.append(final_feat.detach().cpu())

            # Concatenate and build the output entry for current dataset
            y_final = torch.concatenate(y_list)

            if compute_task:
                probas_final = torch.softmax(torch.concatenate(probas), 1)
                task_output.update(
                    {
                        name: {
                            "y": y_final,
                            "probas": probas_final,
                        }
                    }
                )
            if compute_encoder:
                encoder_output_entry = {
                    "y": y_final
                }

                if feat_mode == "all":
                    for k in all_features:
                        all_features[k] = torch.concatenate(all_features[k])
                    encoder_output_entry["feats_by_layer"] = all_features
                if feat_mode == "early" or feat_mode == "final":
                    encoder_output_entry["feats"] = torch.concatenate(encoder_feats)
                if feat_mode == "early":
                    encoder_output_entry["early_feats"] = torch.concatenate(encoder_early_feats)

                encoder_output[name] = encoder_output_entry

                # Memory Cleanup
                del y_list, probas, encoder_feats, encoder_early_feats
                for k in list(all_features.keys()):
                    del all_features[k]
                del all_features
                gc.collect()
                torch.cuda.empty_cache()


        if compute_encoder:
            with open(str(encoder_filename), "wb") as fp:
                pickle.dump(encoder_output, fp)
                print("Encoder dictionary saved successfully to file")
        if compute_task:
            with open(str(model_filename), "wb") as fp:
                pickle.dump(task_output, fp)
                print("Task dictionary saved successfully to file")

        # Memory Cleanup
        if 'model' in locals():
            del model
        if 'encoder' in locals():
            del encoder
        torch.cuda.empty_cache()

    return task_output, encoder_output
