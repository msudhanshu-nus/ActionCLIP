# Code for "ActionCLIP: A New Paradigm for Action Recognition"
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
import clip


def text_prompt(data):
    """
    Build text prompts for CLIP given a dataset object `data` with `data.classes`.

    This version is tailored for the MultiBypass IAE setting with three classes:
    bleeding, mechanical injury, thermal injury.

    For each class, we use 7 detailed, surgery-specific prompts (provided by user).
    The function remains API-compatible with the original ActionCLIP text_prompt:
    it returns (classes, num_text_aug, text_dict).
    """

    # data.classes is usually like:
    # [(0, 'bleeding'), (1, 'mechanical injury'), (2, 'thermal injury')]
    class_names = [c for (_, c) in data.classes]
    lower_names = [c.lower() for c in class_names]

    # --- define your 7 prompts per adverse event ---

    bleeding_prompts = [
        "A video of intraoperative active bleeding during laparoscopic surgery.",
        "Surgical site obscured by pooling red blood.",
        "Active leaking of blood from a tissue dissection site.",
        "Laparoscopic gastric bypass video showing active bleeding at the dissection site.",
        "Laparoscopic view with surgical instruments and fresh red blood oozing from tissue.",
        "Bleeding intraoperative adverse event where exposed vessel and surrounding field appear bright red.",
        "Laparoscopic scene with pooled blood partially obscuring the operative area during dissection.",
    ]

    thermal_prompts = [
        "A video of intraoperative thermal injury caused by electrosurgical instruments.",
        "Sudden generation of surgical smoke obscuring the visual field.",
        "A laparoscopic view of tissue burning and smoke plume generation.",
        "Energy device causing unintended heat spread leading to localized tissue burn.",
        "Thermal damage marked by white-blanched or blackened tissue near the dissection site.",
        "Thermal injury where cautery causes charring or darkened tissue.",
        "Laparoscopic frame showing tissue burn from energy device contact.",
    ]

    mechanical_prompts = [
        "Laparoscopic surgical video showing tissue tearing caused by instrument traction.",
        "Mechanical injury where the grasper accidentally crushes or pinches soft tissue.",
        "A video of accidental mechanical injury caused by surgical graspers or scissors.",
        "Inadvertent tearing or laceration of visceral tissue due to excessive traction.",
        "Instrument-induced tissue damage due to excessive pulling or compression.",
        "Intraoperative mechanical injury marked by a ripped or lacerated tissue edge.",
        "Laparoscopic scene with unintended blunt trauma from surgical tool pressure.",
    ]

    # Map normalized class names to their prompt banks
    prompt_bank = {
        "bleeding": bleeding_prompts,
        "thermal injury": thermal_prompts,
        "mechanical injury": mechanical_prompts,
    }

    def get_class_key(name: str) -> str:
        """Map a class name to one of the keys in prompt_bank using substrings."""
        n = name.lower()
        if "bleed" in n:
            return "bleeding"
        if "thermal" in n:
            return "thermal injury"
        if "mechan" in n:
            return "mechanical injury"
        return None  # unexpected class name

    # We know you defined 7 prompts per class
    num_text_aug = 7

    text_dict = {}
    for aug_idx in range(num_text_aug):
        tokens_per_class = []

        for _, cname in data.classes:
            key = get_class_key(cname)
            if key is not None and key in prompt_bank:
                prompts = prompt_bank[key]
                # just in case: use modulo, but here len(prompts) == 7
                prompt_text = prompts[aug_idx % len(prompts)]
            else:
                # Fallback: if some unexpected class name appears, use a simple generic prompt
                prompt_text = f"A video of {cname}."

            tokens_per_class.append(clip.tokenize(prompt_text))

        # shape: [num_classes, token_length]
        text_dict[aug_idx] = torch.cat(tokens_per_class, dim=0)

    # Concatenate all augmentations over classes
    # classes shape: [num_text_aug * num_classes, token_length]
    classes = torch.cat([v for _, v in text_dict.items()], dim=0)

    return classes, num_text_aug, text_dict
