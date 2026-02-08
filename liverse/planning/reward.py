import torch
import torch.nn.functional as F


def cosine_similarity_reward(world_model, feat, state, action, target_embedding):
    """Compute cosine similarity between predicted embedding and target."""
    predicted_embedding = world_model.heads["reward"](feat)
    return F.cosine_similarity(predicted_embedding, target_embedding, dim=-1)
