"""Experimental encoder feature calibration; physical actor observations stay real."""
import torch


def calibrated_history(model, history, scale):
    value = history.clone()
    indices = [0, 1, 2, 12, 13, 14, 15]
    q, target = history[..., :16], history[..., 16:]
    proxy = target + scale*(q-target)
    proxy = torch.maximum(torch.minimum(proxy, model.upper), model.lower)
    value[..., indices] = proxy[..., indices]
    return value


@torch.no_grad()
def compensated_action(model, obs96, history, scale, release_mask=None):
    if scale == 1.0 and release_mask is None:
        action, logits, clock, _ = model.predict(obs96, history)
        return action, torch.cat((logits.softmax(-1), clock), -1)
    encoder_obs = calibrated_history(model, obs96.reshape(-1, 3, 32), scale).reshape(-1, 96)
    encoder_history = calibrated_history(model, history, scale)
    _, logits, clock, latent = model.predict(encoder_obs, encoder_history)
    command = torch.cat((logits.softmax(-1), clock), -1)
    if release_mask is not None:
        command[release_mask, :4] = 0
        command[release_mask, 1] = 1
    # Only adaptation features change. Actual joint observations enter the
    # frozen motor actor; sent targets/history are never falsified in the loop.
    normalized = ((torch.cat((obs96, command), -1)-model.obs_mean)/torch.sqrt(model.obs_var+1e-5)).clamp(-5, 5)
    raw = model.actor(torch.cat((normalized, latent), -1))
    action = (raw.clamp(-1, 1)*(0.05/model.action_scale)).clamp(-1, 1)*model.action_mask
    return action, command
